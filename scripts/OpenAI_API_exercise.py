import getpass
import openai
from tqdm import tqdm
import json
import os
import pandas as pd
from IPython.display import Markdown, display
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from llama_index import download_loader
from llama_index import (GPTVectorStoreIndex, LLMPredictor, ServiceContext,
                         StorageContext, VectorStoreIndex, download_loader,
                         load_index_from_storage)

# API key
apikey = getpass.getpass(prompt = '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')
apikey = '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'
openai.api_key = apikey
# llama index用に環境変数設定
os.environ["OPENAI_API_KEY"] = apikey

### チャットしてみる
def get_chat_response(messages, model="gpt-3.5-turbo-0613"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response

messages = [{"role": "system", "content": "あなたは親切な人工知能です"},
            {'role':'user','content':'GPTを使いこなすために重要なことは何ですか？'}]

response = get_chat_response(messages, model="gpt-3.5-turbo-0301")
print(response.choices[0].message["content"])
# 使用したトークン数
response["usage"]["total_tokens"]

# temperature : 0.0 ~ 2.0
# 低い：確率値の差分が大きくなり、出力のランダム性が減る
# 高い：確率値の差分が小さくなり、出力のランダム性が増える

# top_p : 0.0 ~ 1.0
# 確率が上位○○%までのトークンのみが候補となる（×100が%の値）

# temperatureとtop_pの併用は非推奨


### プロンプトを入力して返答を返す関数を定義
def get_chatgpt_response(
    user_input: str,
    template: str,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 0,
    max_tokens: int = 500,
):
    """
    ChatGPTに対して対話を投げかけ、返答を取得する
    """
    prompt = template.format(user_input=user_input)
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]
    
PROMPT_TEMPLATE = """
    下記の文章を英語にしてください。

    {user_input}

    """   

print(get_chatgpt_response("がむしゃらにやれ！", PROMPT_TEMPLATE))


### 問い合わせ分類
PROMPT_TEMPLATE = """
    下記の####で区切られた顧客からの問い合わせがあります。
    問い合わせ内容を最初に第一カテゴリーで分けて、その後、第一カテゴリーに紐づく第二カテゴリーに分けてください。
    カテゴリーに当てはまるものが無い場合「該当するカテゴリーがありません」と返答してください。

    第一カテゴリー：
      製品, 注文, 配送

    第二カテゴリー：
     ・製品の場合
       スペック, 値段, 購入方法
     ・注文の場合
       注文状況, 注文変更, 支払い方法
      ・配送の場合
       配送ステータス, 配送オプション, 配送問題

    ####

    {user_input}

    ####

    出力は下記の形式で出力してください
      第一カテゴリー:<第一カテゴリーのどれか>
      第二カテゴリー:<第二カテゴリーのどれか>
    """

get_chatgpt_response("到着日時を変更して欲しいです", PROMPT_TEMPLATE)

# json形式で出力
PROMPT_TEMPLATE = """
    下記の####で区切られた顧客からの問い合わせがあります。
    問い合わせ内容を最初に第一カテゴリーで分けて、その後、第一カテゴリーにそれぞれ紐づく第二カテゴリーに分けてください。
    カテゴリーに当てはまるものが無い場合「該当するカテゴリーがありません」と返答してください。

    第一カテゴリー：
      製品, 注文, 配送

    第二カテゴリー：
     ・製品の場合
       スペック, 値段, 購入方法
     ・注文の場合
       注文状況, 注文変更, 支払い方法
      ・配送の場合
       配送ステータス, 配送オプション, 配送問題

    ####

    {user_input}

    ####

    出力は必ず、JSON形式(key=<第一カテゴリーのどれか>, value=<第二カテゴリーのどれか>)で出力してください。
    例： {{"key": "製品", "value": "値段"}}
    該当するカテゴリーが無い場合key,value両方とも 該当なし としてください
    """

print(get_chatgpt_response("このスマートフォンの最新の価格は何ですか？", PROMPT_TEMPLATE))


### 問い合わせ内容が膨大にある
questions = [
    "このカメラの最大シャッタースピードは何ですか？",
    "私のパッケージの追跡番号を教えてください。",
    "エクスプレス配送は可能ですか？",
    "特定の日時に配送することはできますか？",
    "配送された商品が損傷していました。どうすればいいですか？",
    "注文の支払いにApple PayやGoogle Payは使用できますか？",
    "私のパッケージはいつ到着予定ですか？",
    "注文の支払いにクレジットカードは使用できますか？",
    "私の注文はまだ出荷されていませんか？",
    "この製品はオンラインで注文することができますか？",
    "私のパッケージが遅延しています。なぜですか？",
    "このスマートフォンの最新の価格は何ですか？",
    "注文した商品をキャンセルすることは可能ですか？",
    "商品の配送はどの運送会社を利用していますか？",
    "このサブスクリプションサービスの年間費用は何ですか？",
    "このヘッドフォンの周波数レンジは何ですか？",
    "私の地域への配送は可能ですか？",
    "注文の支払いにビットコインは使用できますか？",
    "配送された商品が間違っていました。どうすればいいですか？",
    "この製品を店舗で購入することは可能ですか？",
    "パッケージが届かない場合、どうすればいいですか？",
    "私の注文はキャンセルされましたか？",
    "このパソコンのRAMの容量はどれくらいですか？",
    "注文の支払いにデビットカードは使用できますか？",
    "私のパッケージは出荷されましたか？",
    "注文の数量を増やすことは可能ですか？",
    "私の注文はすでに確認されましたか？",
    "注文の進行状況を確認するにはどうすればよいですか？",
    "配送先を注文後に変更することは可能ですか？",
    "パッケージが開封されていました。どうすればいいですか？",
    "配達員が不在票を残していました。どうすればいいですか？",
    "私のパッケージは配送中に失われたようです。どうすればいいですか？",
    "この自転車のフレームは何で作られていますか？",
    "この製品を予約注文することは可能ですか？",
    "注文した商品の色やサイズを変更することは可能ですか？",
    "この製品の月々の費用はどれくらいですか？",
    "このスマートフォンのプロセッサーは何ですか？",
    "この製品は分割払いで購入できますか？",
    "この製品は海外から注文できますか？",
    "注文した商品の在庫状況を教えてください。",
    "このゲーム機の割引価格はありますか？",
    "私の注文の配送先住所を変更することは可能ですか？",
    "注文した商品を別のものに変更することは可能ですか？",
    "この洗濯機には何年間の保証が含まれていますか？そのコストはどれくらいですか？",
    "注文の支払いにPayPalは使用できますか？",
]

def get_responses_and_check_json(questions):
    """
    ChatGPTからの応答を取得し、それらがJSON形式であることを確認する
    """
    list_json = []

    def json_checker(response):
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            print(f"No json format... {response.strip()}")

    for question in tqdm(questions):
        response = get_chatgpt_response(question, PROMPT_TEMPLATE)
        list_json.append(json_checker(response))

    return list_json

list_json = get_responses_and_check_json(questions)

df = pd.DataFrame(list_json)
df["question"] = questions
df.columns = ["category_1", "category_2", "question"]
print(df)


### 各カテゴリーに対応した返答文の生成
def get_chatgpt_response(
    user_input: str,
    category_1: str,
    category_2: str,
    template: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: int = 500,
):
    """
    ChatGPTに対して対話を投げかけ、返答を取得する
    """
    prompt = template.format(
        user_input=user_input, category_1=category_1, category_2=category_2
    )
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

PROMPT_TEMPLATE = """
    あなたはカスタマーセンターの従業員です。
    あなたの仕事は顧客の問い合わせに対して、適切な部門の連絡先を教えることです。
    顧客の問い合わせ内容とカテゴリ－は下記の####で区切られた内容です。

    ####
    {user_input}

    第一カテゴリー:{category_1}
    第二カテゴリー:{category_2}
    ####

    下記ステップに従って、メールを作成してください。

    1. 第一カテゴリーと第二カテゴリーに応じてメールアドレスを下記から抽出してください。
        製品：スペック は product_spec@example.com
        製品：値段 は product_price@example.com
        製品：購入方法 は product_purchase_method@example.com
        注文：注文状況 は order_status@example.com
        注文：注文変更 は order_change@example.com
        注文：支払い方法 は order_payment_method@example.com
        配送：配送ステータス は delivery_status@example.com
        配送：配送オプション は delivery_option@example.com
        配送：配送問題 は delivery_issue@example.com

    2. メール文章を作成
      ユーザーからの問い合わせ内容に対して1で抽出したメールアドレスに送信するようにお願いする文章を書いてください。
        形式：
          適切な件名

          文章
    """

for values in df.values[:1]:
    print("")
    print("#" * 100)
    category_1, category_2, user_question = values[0], values[1], values[2]
    print(f"問い合わせ内容:{user_question}\n")
    response = get_chatgpt_response(
        user_input=user_question,
        category_1=category_1,
        category_2=category_2,
        template=PROMPT_TEMPLATE,
    )
    print(response)


### Function Calling
# 外部の関数を定義
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": 25,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# 関数の説明を記載
my_functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "unit"],
        },
    }
]

# 関数への入力を作成する
response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613",
                                        messages=[{"role": "user", "content": "東京の天気は何でしょうか?"}],
                                        functions=my_functions,
                                        function_call="auto",
                                        )

print(response["choices"][0]["message"].get("function_call"))

# json形式に変換：これが関数への入力となる
json_response = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
# 中身を確認
print(json_response["unit"])

# 関数にjsonを渡す
function_response = get_current_weather(location=json_response["location"], unit=json_response["unit"])

function_name = response["choices"][0]["message"]["function_call"]["name"]

message = response["choices"][0]["message"]

# 関数の結果を使って回答を生成させる
second_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": "東京の天気は何でしょうか?"},
        message,
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        },
    ],
)

print(second_response["choices"][0]["message"]["content"])



### PDFからQ&Aを作成
PDFReader = download_loader("PDFReader")

loader = PDFReader()
documents = loader.load_data(file=Path("c:/Users/tomoh/Downloads/コンプライアンスのすべて.pdf"))

service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
)

# indexを作成
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="./storage/")

query_engine = index.as_query_engine()

response = query_engine.query("AIとコンプライアンスについて教えて")
print(response)
print((response.source_nodes[0].node.get_text()))



