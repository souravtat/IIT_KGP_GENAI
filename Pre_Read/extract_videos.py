import requests
import re
import json
import time
from urllib.parse import quote

SESSION_COOKIE = "<ADD_YOURS>"
BASE_URL = "https://ai4icps-upskilling.in/mod/url/view.php?id={}"

VIDEO_IDS = [
    399, 403, 404, 406, 408, 411, 412, 415, 416, 418, 419, 421,
    424, 425, 427, 430, 432, 435, 436, 438, 439, 440, 443, 444,
    446, 447, 449, 450, 453, 454, 456, 457, 458, 461, 463, 466,
    467, 468, 471, 473, 474, 477, 478
]

LECTURE_NAMES = {
    399: "01_Introduction_to_AI",
    403: "02_Math_Foundations_AI_ML_I_Part1",
    404: "03_Math_Foundations_AI_ML_I_Part2",
    406: "04_Math_Foundations_AI_ML_II",
    408: "05_Hands_on_Python_I_Notebooks",
    411: "06_Hands_on_Python_I_Part1",
    412: "07_Hands_on_Python_I_Part2",
    415: "08_Linear_Models_Regression_Classification_Part1",
    416: "09_Linear_Models_Regression_Classification_Part2",
    418: "10_Supervised_ML_I_Bayes_KNN_SVM_Part1",
    419: "11_Supervised_ML_I_Bayes_KNN_SVM_Part2",
    421: "12_Hands_on_ML_I_Supervised_ScikitLearn",
    424: "13_Supervised_ML_II_DecisionTree_RandomForest_Part1",
    425: "14_Supervised_ML_II_DecisionTree_RandomForest_Part2",
    427: "15_Hands_on_ML_II_Classification_Regression_ScikitLearn",
    430: "16_Unsupervised_ML_Clustering_KMeans_PCA",
    432: "17_Hands_on_ML_III_Unsupervised_ScikitLearn",
    435: "18_Neural_Networks_DL_FeedForward_CNN_RNN_Part1",
    436: "19_Neural_Networks_DL_FeedForward_CNN_RNN_Part2",
    438: "20_DL_Models_Inference_Pretrained_Pytorch_Part1",
    439: "21_DL_Models_Inference_Pretrained_Pytorch_Part2",
    440: "22_DL_Models_Inference_Pretrained_Pytorch_Part3",
    443: "23_DL_Transfer_Learning_Pretrained_Part1",
    444: "24_DL_Transfer_Learning_Pretrained_Part2",
    446: "25_DL_Applications_Computer_Vision_Part1",
    447: "26_DL_Applications_Computer_Vision_Part2",
    449: "27_Hands_on_DL_Computer_Vision_Part1",
    450: "28_Hands_on_DL_Computer_Vision_Part2",
    453: "29_Introduction_to_NLP_Part1",
    454: "30_Introduction_to_NLP_Part2",
    456: "31_Hands_on_NLP_Part1",
    457: "32_Hands_on_NLP_Part2",
    458: "33_Hands_on_NLP_Part3",
    461: "34_Advanced_DL_Transformers_BERT",
    463: "35_Advanced_DL_Lab_Transformers_BERT",
    466: "36_Prompting_OpenSource_LLMs_Llama2_Mistral_Part1",
    467: "37_Prompting_OpenSource_LLMs_Llama2_Mistral_Part2",
    468: "38_Prompting_OpenSource_LLMs_Llama2_Mistral_Part3",
    471: "39_Foundations_of_Generative_AI",
    473: "40_LLM_Finetuning_Agentic_Workflows_Part1",
    474: "41_LLM_Finetuning_Agentic_Workflows_Part2",
    477: "42_Application_Lab_CV_GANs_VAEs_Part1",
    478: "43_Application_Lab_CV_GANs_VAEs_Part2",
}

cookies = {"MoodleSession": SESSION_COOKIE}
results = {}

for vid in VIDEO_IDS:
    url = BASE_URL.format(vid)
    name = LECTURE_NAMES.get(vid, f"lecture_{vid}")
    try:
        resp = requests.get(url, cookies=cookies, timeout=30)
        # Extract full URLs from within quotes (handles spaces)
        mp4_matches = re.findall(r'"(https?://[^"]+\.mp4)"', resp.text)
        
        video_url = None
        if mp4_matches:
            video_url = mp4_matches[0]
        
        # URL-encode the path part (handle spaces)
        if video_url:
            parts = video_url.split("/walmart/")
            if len(parts) == 2:
                encoded_url = parts[0] + "/walmart/" + quote(parts[1])
            else:
                encoded_url = video_url.replace(" ", "%20")
        else:
            encoded_url = None
            
        results[name] = {"id": vid, "video_url": video_url, "download_url": encoded_url}
        status = "OK" if video_url else "NO VIDEO FOUND"
        print(f"[{status}] {name}")
        if video_url:
            print(f"         {encoded_url}")
    except Exception as e:
        results[name] = {"id": vid, "video_url": None, "error": str(e)}
        print(f"[ERROR] {name}: {e}")
    time.sleep(0.5)

with open("/Users/u0j006f/Projects/ai4icps-notes/video_urls.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Found {sum(1 for v in results.values() if v.get('video_url'))} videos out of {len(VIDEO_IDS)} pages.")
