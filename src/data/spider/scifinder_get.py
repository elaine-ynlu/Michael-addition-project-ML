import httpx
import json
import os
import time
import sys

# --- 配置参数 ---
BASE_URL = "https://scifinder-n.cas.org/internal/api/export/reaction/rdf"
TOTAL_RECORDS = 312584  # 从 URL 参数 totalAnswerCount 获取
CHUNK_SIZE = 500       # 每次请求导出的记录数量 (与示例一致)
MAX_RETRIES = 3        # 最大重试次数
RETRY_DELAY = 10       # 重试间隔时间 (秒)
REQUEST_DELAY = 2      # 每次成功请求后的基本延迟 (秒)，避免过快请求

OUTPUT_DIR = "rdf_exports"  # RDF 文件保存目录
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.txt") # 进度记录文件

# 从你提供的 URL 中提取的固定参数
URL_PARAMS = {
    "totalAnswerCount": TOTAL_RECORDS,
    # "uiContext": "366",
    # "uiSubContext": "361",
    # "appId": "60ae3e7b-bf91-4ce4-bf65-15833d9c10f8", # 警告：这个值可能很快失效
    # "sourceInitiatingActionId": "2221d23d-611d-4dc9-b608-d87ffb09121b" # 警告：这个值可能很快失效
}

FACET_FILE_PATH = "facet_data.json"
try:
    with open(FACET_FILE_PATH, 'r', encoding='utf-8') as f:
        FACET_GROUP_LIST_DATA = json.load(f)
except FileNotFoundError:
    print(f"错误: 未找到 facet 数据文件 {FACET_FILE_PATH}")
    print("请将你提供的 facetGroupList JSON 数据保存到该文件中。")
    FACET_GROUP_LIST_DATA = [] # 或者 sys.exit(1)
except json.JSONDecodeError:
    print(f"错误: 解析 facet 数据文件 {FACET_FILE_PATH} 失败。请检查JSON格式。")
    FACET_GROUP_LIST_DATA = [] # 或者 sys.exit(1)


PAYLOAD_TEMPLATE = {
    "query": {"uriList": []},
    "navKey": "682a901e3e83c3220641e4a8", # 警告：这个值可能很快失效
    "requestingSDFile": False,
    "requestingRDFile": True,
    "resultView": "list_of_details",
    "methodStepsToExport": [],
    "facetGroupList": FACET_GROUP_LIST_DATA, # 使用上面定义的完整数据
    "currentDateTime": "20250519_1443", # 这个值可以动态生成，但通常不关键
    "exportOptions": [],
    "exportFileName": "Reaction_20250519_1443", # 这个也会在循环中动态修改
    "exportRange": {"minimum": 1, "maximum": CHUNK_SIZE}, # 将在循环中动态修改
    "exportRangeType": "RANGE",
    "requestingV3000": True
}

minimal_payload = {
    "navKey": PAYLOAD_TEMPLATE["navKey"], # 确保 navKey 有效
    "requestingRDFile": True,
    "exportRange": {
        "minimum": 1,
        "maximum": 499
    },
    "requestingV3000": True
}

HEADERS = {
    'Host': 'scifinder-n.cas.org',
    # 'Connection': 'keep-alive',
    # 'X-Request-Id': '31d51d4f-a831-4040-865a-7e1acdf2bf65',
    # 'X-Ping-Protected': 'True',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-ua': '"Chromium";v="136", "Microsoft Edge";v="136", "Not.A/Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    # 'X-App-Id': '60ae3e7b-bf91-4ce4-bf65-15833d9c10f8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'Accept': 'application/json, text/plain, */*',
    'Content-Type': 'application/json',
    'Origin': 'https://scifinder-n.cas.org',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://scifinder-n.cas.org/',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en,en-GB;q=0.9,en-US;q=0.8,zh-CN;q=0.7,zh;q=0.6'
}

# --- 辅助函数 ---
def load_progress():
    """加载已下载的记录数"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    last_exported, last_filename = content.split(',')
                    print(f"Resuming download. Last successfully exported up to record: {last_exported}")
                    return int(last_exported)
        except Exception as e:
            print(f"Error reading progress file ({PROGRESS_FILE}): {e}. Starting from scratch.")
    return 0

def save_progress(num_exported, filename):
    """保存当前已下载的记录数和最后一个文件名"""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"{num_exported},{filename}")

def make_request(session, start_index, end_index):
    """发送POST请求并处理重试"""
    payload = PAYLOAD_TEMPLATE.copy() # 使用模板的副本
    payload["exportRange"]["minimum"] = start_index
    payload["exportRange"]["maximum"] = end_index
    
    # 动态生成文件名和 payload 中的 exportFileName (可选, 但推荐)
    # 获取当前时间戳，例如 YYYYMMDD_HHMMSS
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    # payload["currentDateTime"] = current_timestamp # 可以更新这个时间
    payload["exportFileName"] = f"Reaction_{current_timestamp}_{start_index}_{end_index}"

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Requesting records {start_index}-{end_index} (Attempt {attempt + 1}/{MAX_RETRIES})...")
            response = session.post(BASE_URL, params=URL_PARAMS, json=minimal_payload, headers=HEADERS, timeout=60) # 60秒超时
            if response.status_code == 302:
                redirect_url = response.headers.get('Location')
                print(f"Received HTTP 302. Redirecting to: {redirect_url}")
                print(f"History of requests: {response.history}")  # 查看请求历史


            if response.status_code == 200:
                # 检查响应内容是否真的是RDF。有时API会返回JSON错误信息但状态码200
                # RDF文件通常以 $RDFILE 开头，或包含 <rdf:RDF> 等XML/RDF特定标签
                # SciFinder的RD文件通常不是XML RDF，而是另一种文本格式。
                # 简单的检查可以是内容是否为空，或者是否有特定的错误信息。
                if response.content and not response.json().get("error"): # 假设错误会以JSON形式返回
                     print(f"Successfully fetched records {start_index}-{end_index}.")
                     return response.content
                else:
                    print(f"Error in response content for {start_index}-{end_index}. Status: {response.status_code}. Response: {response.text[:200]}...")
                    # 如果返回的是JSON错误，可以尝试解析并打印
                    try:
                        error_data = response.json()
                        print(f"API Error: {error_data}")
                        if "message" in error_data and "session" in error_data["message"].lower():
                            print("Session might have expired. Please update navKey, appId, etc.")
                            return None # 不再重试会话错误
                    except json.JSONDecodeError:
                        pass # 不是JSON错误

            else:
                print(f"Error fetching records {start_index}-{end_index}. Status: {response.status_code}. Response: {response.text[:200]}...")
            
            if response.status_code in [401, 403]: # 未授权或禁止访问，通常意味着会话/token问题
                print("Authorization error. Halting. Please check your appId, navKey, and session.")
                return None # 不再重试授权错误

        except httpx.RequestError as e:
            print(f"Request exception for {start_index}-{end_index}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY * (attempt + 1)) # 指数退避或固定延迟
        else:
            print(f"Failed to fetch records {start_index}-{end_index} after {MAX_RETRIES} attempts.")
            return None
    return None


# --- 主逻辑 ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    exported_count = load_progress()
    start_from = exported_count + 1

    if start_from > TOTAL_RECORDS:
        print("All records seem to be already downloaded according to progress file.")
        return

    print(f"Starting download from record {start_from} up to {TOTAL_RECORDS}.")
    
    proxy_url = "http://127.0.0.1:7890"  # 适配 httpx，直接传递字符串即可

    with httpx.Client(proxy=proxy_url) as session:  # 注意参数名为proxies，且可直接用字符串

        for i in range(start_from, TOTAL_RECORDS + 1, CHUNK_SIZE):
            current_min = i
            current_max = min(i + CHUNK_SIZE - 1, TOTAL_RECORDS)

            rdf_data = make_request(session, current_min, current_max)

            if rdf_data:
                file_name = f"reactions_{current_min}_{current_max}.rdf"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                try:
                    with open(file_path, 'wb') as f: # RDF文件可能是二进制或特定编码文本
                        f.write(rdf_data)
                    print(f"Saved: {file_path}")
                    exported_count = current_max
                    save_progress(exported_count, file_name)
                    
                    # 打印进度
                    progress_percentage = (exported_count / TOTAL_RECORDS) * 100
                    sys.stdout.write(f"\rProgress: {exported_count}/{TOTAL_RECORDS} ({progress_percentage:.2f}%) chunks completed.  ")
                    sys.stdout.flush()

                except IOError as e:
                    print(f"Error writing file {file_path}: {e}")
                    print("Stopping download.")
                    return # 严重错误，停止
                
                time.sleep(REQUEST_DELAY) # 在每次成功请求后暂停一下
            else:
                print(f"Failed to download chunk {current_min}-{current_max}. Stopping.")
                # 你可以选择在这里 continue 来跳过失败的块，但对于顺序下载通常选择停止
                return 

    print(f"\nDownload completed. Total records processed: {exported_count}")


if __name__ == "__main__":
    # 确保你的 FACET_GROUP_LIST_DATA 变量包含了你提供的完整的 facetGroupList JSON 数据
    if not FACET_GROUP_LIST_DATA or not isinstance(FACET_GROUP_LIST_DATA, list) or len(FACET_GROUP_LIST_DATA[0].get("binConstraints", [])) < 10:
         print("错误：FACET_GROUP_LIST_DATA 似乎不完整或未正确设置。")
         print("请将你提供的完整 facetGroupList JSON 数据赋值给 FACET_GROUP_LIST_DATA 变量。")
         # 示例性地填充了一小部分，你需要替换成你完整的 facetGroupList
         # 此处仅为防止直接运行时因 FACET_GROUP_LIST_DATA 为空而报错，实际使用时务必替换！
         if not FACET_GROUP_LIST_DATA:
             print("FACET_GROUP_LIST_DATA 为空，脚本可能无法按预期工作。")

    main()
