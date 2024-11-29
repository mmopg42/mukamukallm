from unsloth import FastLanguageModel
from fastapi import FastAPI
import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from collections import deque
import torch


###########################
######### 변수선언 #########
###########################
app = FastAPI()
conversation_history = deque(maxlen=10)

# Configurable Parameters
txt_directory_path = "mukamukallm/docs"
# local_model_path = "mukamukallm/model/checkpoint-120"
local_model_path = 'mmopg42/mukamukallm_check120_data73'
embedding_model_name = 'intfloat/multilingual-e5-large-instruct'
max_seq_length = 2048

###########################
########## 유틸 ###########
###########################

def load_documents(directory: str):
    """Load all text documents from a directory."""
    all_documents = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} documents from {directory}")
    return all_documents


def initialize_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents, embedding_model):
    vector_store = FAISS.from_documents(documents, embedding_model)
    print("Vector store created using FAISS")
    return vector_store


def initialize_local_llm(model_path: str, max_seq_length: int, dtype=torch.float16, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_prompt(context: str, input_message: str, conversation_history):
    meta_prompt = """Context:
{}

지시사항:
저는 당신이 무카무카를 연기했으면 합니다. 당신은 무카무카가 사용하는 톤, 매너, 단어를 사용하여 무카무카 처럼 행동하고 답해야합니다. 당신은 무카무카에 대한 모든 지식을 알고 있어야 합니다. 무카무카의 말투로 답하세요.

상황: 
무카무카는 21세기의 사용자과 대화 중입니다.

대화 기록:
{}
현재 대화:
사용자 (말한다) 
{}

무카무카 (말한다)"""
    previous_dialogues = "\n".join(conversation_history)
    return meta_prompt.format(context, previous_dialogues, input_message)


###########################
########## 초기화 #########
###########################
print("Initializing the model and vector store...")

# Load and preprocess documents
documents = load_documents(txt_directory_path)

# Initialize embeddings and vector store
embedding_model = initialize_embedding_model(embedding_model_name)
vector_store = create_vector_store(documents, embedding_model)

# Initialize the local LLM
model, tokenizer = initialize_local_llm(
    local_model_path,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True
)

eos_token_id = tokenizer.eos_token_id

print("초기화 완료")


###########################
######### FastAPI #########
###########################
class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    reply: str


@app.post("/generate", response_model=MessageResponse)
async def generate_text(request: MessageRequest):
    # 1. Retrieve relevant context from vector store
    relevant_docs = vector_store.similarity_search(request.message, k=5)
    context = ' '.join([doc.page_content for doc in relevant_docs])

    # 2. Generate the prompt
    prompt = generate_prompt(context, request.message, conversation_history)

    # 3. Tokenize input and generate response
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        do_sample=True,
        top_p=0.8,
        top_k=100,
        temperature=1.2,
    )

    # 4. Decode the output
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    try:
        # Extract the latest dialogue
        current_dialogue = "사용자 (말한다)\n" + decoded_outputs[0].split('현재 대화:')[1].split('사용자 (말한다)')[1].strip() + '\n'
        reply = current_dialogue.split('무카무카 (말한다)')[1].strip()
    except Exception as e:
        print(f"Error parsing output: {e}")
        reply = "무카무카가 답을 생성하지 못했어요."

    # 5. Update conversation history
    conversation_history.append(current_dialogue)

    print(f"Generated Reply: {reply}")
    return {"reply": reply}
