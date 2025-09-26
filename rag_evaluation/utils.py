import json
import os

def read_all_files_to_bytes(folder_path, file_extensions=None):
    """
    讀取指定資料夾中指定類型的檔案，並將其內容轉換為位元組格式。

    Args:
        folder_path (str): 包含檔案的資料夾路徑。
        file_extensions (list): 要讀取的檔案副檔名列表，如 ['.pdf', '.md', '.txt']。
                               如果為 None，則預設讀取 PDF 和 Markdown 檔案。

    Returns:
        dict: 字典，鍵為檔案名稱，值為檔案的位元組內容。
              如果資料夾不存在或沒有指定類型的檔案，則返回空字典。
    """
    # 預設支援的檔案類型
    if file_extensions is None:
        file_extensions = ['.pdf', '.md', '.markdown']
    
    # 將副檔名轉為小寫以便比較
    file_extensions = [ext.lower().strip() for ext in file_extensions]
    
    # 建立一個空字典來存放結果
    file_contents = {}

    # 檢查路徑是否存在
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"錯誤: 資料夾 '{folder_path}' 不存在。")

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        # 跳過目錄
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue
            
        # 取得檔案副檔名
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 檢查檔案是否為指定的類型
        if file_ext in file_extensions:
            try:
                # 以二進位讀取模式打開並讀取檔案
                with open(file_path, 'rb') as file:
                    content_bytes = file.read()
                    # 將結果存入字典
                    file_contents[filename] = content_bytes
                print(f"已讀取檔案: {filename} ({file_ext}, {len(content_bytes)} bytes)")
            except Exception as e:
                print(f"讀取檔案 '{filename}' 時發生錯誤: {e}")

    print(f"共讀取 {len(file_contents)} 個檔案")
    return file_contents
