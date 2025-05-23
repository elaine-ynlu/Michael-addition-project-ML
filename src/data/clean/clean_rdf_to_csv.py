import os
import glob
import rdflib
import pandas as pd
from src.data.common.paths import project_paths
from rdkit import Chem
from rdkit.Chem import rdmolops
import re
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mol_block_to_smiles(mol_block):
    """
    将 MOL 格式的文本块转换为 SMILES 字符串，主要步骤：
    1. 去除 $MOL 标记和纯分子式行
    2. 定位 V2000 计数行；若未找到则尝试 V3000 或严格解析
    3. 构建标准 Molfile 块（保证有 3 行头部）
    4. 调用 RDKit 解析并在必要时手动 Sanitize，忽略 valence 错误
    """
    if not mol_block or not mol_block.strip():
        logger.warning("输入为空或仅包含空白，无法解析。")
        return None

    # 拆分行并去除 $MOL 标记行
    lines = mol_block.splitlines()
    lines = [line for line in lines if not line.strip().startswith("$MOL")]

    # 定义正则：分子式行和 V2000 计数行
    formula_re = re.compile(r'^\s*([A-Z][a-z]?\d*)(\s+([A-Z][a-z]?\d*))*\s*$')
    counts_re  = re.compile(r'^\s*(\d+)\s+(\d+).*V2000')

    # 过滤：去掉空行和纯分子式行
    filtered = []
    for line in lines:
        if not line.strip():
            continue
        if formula_re.match(line) and not counts_re.match(line):
            continue
        filtered.append(line)

    # 定位 V2000 计数行索引
    counts_idx = None
    for i, line in enumerate(filtered):
        if counts_re.match(line):
            counts_idx = i
            break

    # 如果未找到 V2000，尝试 V3000 或 strictParsing
    if counts_idx is None:
        logger.warning("无法在 MOL 块中找到 V2000 计数行，尝试其他解析方式。")
        # 尝试 V3000 并用 sanitize=False
        if any("V3000" in ln for ln in lines):
            logger.info("检测到 V3000 格式，尝试用 sanitize=False 解析原始块。")
            mol_v3 = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
            if mol_v3:
                try:
                    rdmolops.SanitizeMol(
                        mol_v3,
                        sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_ALL ^ rdmolops.SanitizeFlags.SANITIZE_PROPERTIES
                    )
                except Exception as e:
                    logger.warning(f"SanitizeMol 警告: {e}，继续生成 SMILES。")
                smiles = Chem.MolToSmiles(mol_v3)
                logger.info(f"V3000 解析成功，SMILES={smiles}")
                return smiles
            else:
                logger.error("RDKit 无法解析 V3000 Molfile。")

        # strictParsing=True 严格解析后再 sanitize
        header_fb = [""] * 3
        fallback_block = "\n".join(header_fb + filtered)
        logger.info("尝试使用 strictParsing=True 严格解析补齐后的 Molfile。")
        mol_strict = Chem.MolFromMolBlock(fallback_block, removeHs=False, strictParsing=True)
        if mol_strict:
            try:
                rdmolops.SanitizeMol(
                    mol_strict,
                    sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_ALL ^ rdmolops.SanitizeFlags.SANITIZE_PROPERTIES
                )
            except Exception as e:
                logger.warning(f"SanitizeMol 警告: {e}，继续生成 SMILES。")
            smiles = Chem.MolToSmiles(mol_strict)
            logger.debug(f"strictParsing 解析成功，SMILES={smiles}")
            return smiles

        return None

    # 划分头部和主体
    header = filtered[:counts_idx]
    body = filtered[counts_idx:]
    while len(header) < 3:
        header.insert(0, "")
    new_block = "\n".join(header + body)

    # 用 RDKit 解析并忽略 valence 错误
    mol = Chem.MolFromMolBlock(new_block, removeHs=False, sanitize=False)
    if mol:
        try:
            rdmolops.SanitizeMol(
                mol,
                sanitizeOps=rdmolops.SanitizeFlags.SANITIZE_ALL ^ rdmolops.SanitizeFlags.SANITIZE_PROPERTIES
            )
        except Exception as e:
            logger.warning(f"SanitizeMol 警告: {e}，继续生成 SMILES。")
        smiles = Chem.MolToSmiles(mol)
        logger.debug(f"解析成功，SMILES={smiles}")
        return smiles
    else:
        logger.error("RDKit 无法解析清理后的 Molfile。")
        return None

def extract_rdf_to_csv(rdf_filepath, csv_filepath):
    """
    从单个 MDL RDF 文件提取反应数据并保存为 CSV。
    """
    print(f"开始处理文件: {rdf_filepath}")
    all_reactions = []

    try:
        with open(rdf_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件出错 ({rdf_filepath}): {e}")
        return

    blocks = content.split('$RXN')
    if len(blocks) < 2:
        print(f"文件 {rdf_filepath} 中未找到 '$RXN' 分隔的反应数据。")
        return

    for idx, block in enumerate(blocks[1:], start=1):
        print(f"处理反应块 {idx}")
        data = {}
        lines = block.strip().split('\n')

        react_blocks, prod_blocks = [], []
        in_mol, cur_mol = False, []
        mol_count, num_reac, num_prod, last_key = 0, 0, 0, None

        m = re.search(r'\$REREG\s+(\d+)', block)
        if m: num_reac = int(m.group(1))
        m = re.search(r'\$PREG\s+(\d+)', block)
        if m: num_prod = int(m.group(1))

        for ln in lines:
            s = ln.strip()
            if s == '$MOL':
                in_mol = True
                cur_mol = []
                continue
            if in_mol:
                cur_mol.append(ln)
                if s == 'M  END':
                    in_mol = False
                    mol_count += 1
                    mb = "\n".join(cur_mol)
                    if num_reac and len(react_blocks) < num_reac:
                        react_blocks.append(mb)
                    elif num_prod and len(prod_blocks) < num_prod:
                        prod_blocks.append(mb)
                    else:
                        if mol_count == 1:
                            react_blocks.append(mb)
                        else:
                            prod_blocks.append(mb)
                continue

            if s.startswith('$DTYPE'):
                last_key = re.sub(r'\(\d+\)', '', s[len('$DTYPE'):].strip()).strip()
                continue
            if s.startswith('$DATUM') and last_key:
                val = s[len('$DATUM'):].strip()
                if not val:
                    idx_ln = lines.index(ln)
                    if idx_ln + 1 < len(lines):
                        nxt = lines[idx_ln+1].strip()
                        if not nxt.startswith('$'):
                            val = nxt
                if last_key in data:
                    if isinstance(data[last_key], list):
                        data[last_key].append(val)
                    else:
                        data[last_key] = [data[last_key], val]
                else:
                    data[last_key] = val
                last_key = None
                continue

            if s.startswith('$RIREG'):
                data['Reaction_ID'] = s[len('$RIREG'):].strip()

        reac_smiles = [mol_block_to_smiles(mb) for mb in react_blocks]
        prod_smiles = [mol_block_to_smiles(mb) for mb in prod_blocks]
        reac_smiles = [s for s in reac_smiles if s]
        prod_smiles = [s for s in prod_smiles if s]

        data['Reactants_SMILES'] = '.'.join(reac_smiles)
        data['Products_SMILES'] = '.'.join(prod_smiles)
        if reac_smiles or prod_smiles:
            data['Reaction_SMILES'] = f"{data['Reactants_SMILES']}>>{data['Products_SMILES']}"

        if data:
            all_reactions.append(data)
        else:
            print(f"反应块 {idx} 未提取到有效数据。")

    if not all_reactions:
        print(f"文件 {rdf_filepath} 中未提取到任何反应。")
        return

    try:
        df = pd.DataFrame(all_reactions)
        cols_pref = ['Reaction_ID','Reactants_SMILES','Products_SMILES','Reaction_SMILES']
        cols_all = list(df.columns)
        ordered = [c for c in cols_pref if c in cols_all] + [c for c in cols_all if c not in cols_pref]
        df = df[ordered]
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        print(f"已保存到 {csv_filepath}，共 {len(df)} 条记录。")
    except Exception as e:
        print(f"保存 CSV 出错 ({csv_filepath}): {e}")

def process_all_rdf_files():
    """
    从 origin 目录批量处理 RDF 文件，输出到 unclean 目录。
    """
    origin = project_paths.data_origin
    unclean = project_paths.data_unclean
    os.makedirs(unclean, exist_ok=True)
    files = glob.glob(os.path.join(origin, '*.rdf')) + glob.glob(os.path.join(origin, '*.RDF'))
    for f in files:
        out_name = os.path.splitext(os.path.basename(f))[0] + '_extracted.csv'
        out_path = os.path.join(unclean, out_name)
        extract_rdf_to_csv(f, out_path)

if __name__ == "__main__":
    print("开始从 MDL RDF 文件提取并转换为 CSV...")
    process_all_rdf_files()
    print("处理完成。")