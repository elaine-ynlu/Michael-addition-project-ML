import re
from .logger import ProcessLogger

class MetalFilter:
    """金属过滤处理器"""
    def __init__(self):
        self.logger = ProcessLogger('MetalFilter')
        self._init_metal_patterns()

    def _init_metal_patterns(self):
        """初始化金属匹配模式"""
        # 金属符号列表（小写，用于不区分大小写的检查）
        self.metal_symbols_lower = {
            'li', 'na', 'k', 'rb', 'cs', 'fr', 'be', 'mg', 'ca', 'sr', 'ba', 'ra',
            'sc', 'y', 'la', 'ac', 'ti', 'zr', 'hf', 'rf', 'v', 'nb', 'ta', 'db',
            'cr', 'mo', 'w', 'sg', 'mn', 'tc', 're', 'bh', 'fe', 'ru', 'os', 'hs',
            'co', 'rh', 'ir', 'mt', 'ni', 'pd', 'pt', 'ds', 'cu', 'ag', 'au', 'rg',
            'zn', 'cd', 'hg', 'cn', 'al', 'ga', 'in', 'tl', 'nh', 'ge', 'sn', 'pb',
            'fl', 'sb', 'bi', 'mc', 'po', 'lv'
        }

        # 编译正则表达式模式
        self.metal_pattern = re.compile(
            # 匹配带括号的离子：[Symbol+charge], [Symbol], [Symbol-charge]
            r'^\s*\[([a-z]{1,2})\s*([\+\-]\d*)?\]\s*$'
            # 匹配简单元素符号
            r'|\b(Li|Na|K|Rb|Cs|Fr|Be|Mg|Ca|Sr|Ba|Ra|Sc|Y|La|Ac|Ti|Zr|Hf|Rf|V|Nb|Ta|Db|Cr|Mo|W|Sg|Mn|Tc|Re|Bh|Fe|Ru|Os|Hs|Co|Rh|Ir|Mt|Ni|Pd|Pt|Ds|Cu|Ag|Au|Rg|Zn|Cd|Hg|Cn|Al|Ga|In|Tl|Nh|Ge|Sn|Pb|Fl|Sb|Bi|Mc|Po|Lv)\b'
            , re.IGNORECASE
        )

    def is_metal_component(self, component):
        """检查组分是否为金属"""
        if not component:
            return False

        comp_strip = component.strip()
        if not comp_strip:
            return False

        # 使用正则表达式检查
        match = self.metal_pattern.search(comp_strip)
        if match:
            # 如果通过括号组匹配，检查括号内的符号是否为金属
            bracket_symbol = match.group(1)
            if bracket_symbol and bracket_symbol.lower() in self.metal_symbols_lower:
                return True
            # 如果通过简单元素组匹配，直接是金属符号
            elif match.group(2) and match.group(2).lower() in self.metal_symbols_lower:
                return True
            # 处理像 '[H+]' 或 '[Cl-]' 这样的情况
            elif bracket_symbol and bracket_symbol.lower() not in self.metal_symbols_lower:
                return False
            # 如果只有简单元素部分匹配，则是金属
            elif match.group(2):
                return True

        # 备用检查：检查整个组分字符串是否为金属符号
        if comp_strip.lower() in self.metal_symbols_lower:
            return True

        return False

    def remove_metal_components(self, smiles_str):
        """从SMILES字符串中移除金属组分"""
        if not smiles_str:
            return ""

        components = smiles_str.split('.')
        filtered_components = []

        for comp in components:
            if not self.is_metal_component(comp):
                filtered_components.append(comp)

        return '.'.join(filtered_components)

    def filter_reaction_components(self, reaction_parts):
        """过滤反应组分中的金属"""
        if not isinstance(reaction_parts, dict):
            self.logger.error("输入必须是包含反应组分的字典")
            return None, "无效的输入格式"

        filtered_parts = {}
        
        # 过滤反应物
        filtered_parts['reactants'] = self.remove_metal_components(reaction_parts.get('reactants', ''))
        if not filtered_parts['reactants']:
            return None, "过滤后反应物为空"

        # 过滤试剂（如果有）
        agents = reaction_parts.get('agents')
        if agents:
            filtered_parts['agents'] = self.remove_metal_components(agents)
            if not filtered_parts['agents']:
                filtered_parts['agents'] = None
        else:
            filtered_parts['agents'] = None

        # 过滤产物
        filtered_parts['products'] = self.remove_metal_components(reaction_parts.get('products', ''))
        if not filtered_parts['products']:
            return None, "过滤后产物为空"

        return filtered_parts, None 