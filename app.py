from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

# 2. Update API Keys: Load Google API Key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}  # Increased from 3 to 5 for better context retrieval
)

# 3. Initialize Gemini Model with fallback mechanism and actual API test
# Try different model names in order of preference and test each one
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', '').strip()

# Default fallback order: try most stable models first
# Based on Google Gemini API, these are the commonly available models
default_models = [
    'gemini-pro',  # Most stable, widely available (v1) - try this first
    'gemini-1.5-pro',  # 1.5 Pro model
    'gemini-1.5-flash',  # 1.5 Flash model (faster)
    'gemini-2.0-flash-exp',  # Experimental 2.0 Flash
    'gemini-3-pro-preview',  # Latest preview model
]

# When Google migrates model labels (e.g., adds "-latest"), we automatically try both forms
def expand_model_variants(model_name: str):
    """Return possible variant names for a model (e.g., add/remove '-latest')."""
    suffix = "-latest"
    variants = [model_name]
    if model_name.endswith(suffix):
        base_name = model_name[:-len(suffix)]
        if base_name.endswith("-"):
            base_name = base_name[:-1]
        if base_name:
            variants.append(base_name)
    else:
        variants.append(f"{model_name}{suffix}")
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            unique_variants.append(variant)
    return unique_variants

# If user specified a model, try it first, then fall back to defaults
if GEMINI_MODEL:
    model_candidates = [GEMINI_MODEL] + [m for m in default_models if m != GEMINI_MODEL]
    print(f"GEMINI_MODEL environment variable is set to: {GEMINI_MODEL}")
else:
    model_candidates = default_models
    print("No GEMINI_MODEL set, using default model order")

print(f"Will try models in this order: {model_candidates}\n")

chatModel = None
model_used = None
current_model_idx = None


def _format_source_label(metadata):
    if not metadata:
        return None
    title = metadata.get("title")
    source = metadata.get("source")
    if title and source:
        return f"{title} ({source})"
    return source or title


def _build_sources_block(context_value):
    if not isinstance(context_value, list):
        return ""
    seen = set()
    lines = []
    for doc in context_value:
        metadata = getattr(doc, "metadata", None)
        label = _format_source_label(metadata or {})
        if not label or label in seen:
            continue
        seen.add(label)
        lines.append(f"- {label}")
    if not lines:
        return ""
    return "Sources:\n" + "\n".join(lines)


def list_available_models():
    """List all available models using google-genai SDK"""
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        models = client.models.list()
        return [model.name.split('/')[-1] for model in models if hasattr(model, 'name')]
    except Exception as e:
        print(f"  → Could not list available models: {str(e)[:100]}")
        return None

def test_model_with_google_genai(model_name):
    """Test model using google-genai SDK directly (faster and more reliable)
    Returns: True if model works, 'quota' if model exists but quota exceeded, False if not found, None if SDK not available
    """
    try:
        from google import genai
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # Try to generate content - this will fail fast if model doesn't exist
        response = client.models.generate_content(
            model=model_name,
            contents="Hi",
        )
        # If we get here, the model works
        return True
    except ImportError:
        # google-genai not installed, fall back to langchain testing
        return None
    except Exception as e:
        error_str = str(e).lower()
        error_msg = str(e)
        
        # Check for model availability errors - fail fast
        if any(keyword in error_str for keyword in ["not found", "404", "not supported", "not available"]):
            print(f"  → Model {model_name} is not available: {error_msg[:150]}")
            return False
        
        # Quota errors mean the model exists but quota is exceeded - treat as valid model
        if "quota" in error_str or "429" in error_msg or "resourceexhausted" in error_str:
            print(f"  → Model {model_name} exists but quota exceeded (will use model anyway): {error_msg[:100]}")
            return 'quota'  # Special return value to indicate quota issue but model exists
        
        # For other errors, assume model might be valid
        print(f"  → Model {model_name} test warning: {error_msg[:150]}")
        return True

def test_model(model_instance, model_name):
    """Test if the model actually works by making a simple API call with timeout"""
    # First try using google-genai SDK directly (faster)
    google_genai_result = test_model_with_google_genai(model_name)
    if google_genai_result is not None:
        return google_genai_result
    
    # Fall back to langchain testing if google-genai not available
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def invoke_model():
        """Run the model invoke in a separate thread"""
        try:
            test_response = model_instance.invoke("Hi")
            result_queue.put(True)
        except Exception as e:
            exception_queue.put(e)
    
    # Start the model call in a separate thread
    thread = threading.Thread(target=invoke_model, daemon=True)
    thread.start()
    thread.join(timeout=5)  # Reduced timeout to 5 seconds for faster failure
    
    # Check if we got an exception first (this might have happened quickly)
    if not exception_queue.empty():
        e = exception_queue.get()
        error_str = str(e).lower()
        error_msg = str(e)
        
        # Check for model availability errors - fail fast if we see these
        if any(keyword in error_str for keyword in ["not found", "404", "not supported", "not available"]):
            print(f"  → Model {model_name} is not available: {error_msg[:150]}")
            return False
        
        # Quota errors mean the model exists but quota is exceeded - treat as valid model
        if "quota" in error_str or "429" in error_msg or "resourceexhausted" in error_str:
            print(f"  → Model {model_name} exists but quota exceeded (model is valid): {error_msg[:100]}")
            return 'quota'  # Return special value to indicate quota issue but model exists
        
        # For other errors, assume model might be valid
        print(f"  → Model {model_name} test warning (model may be valid): {error_msg[:150]}")
        return True
    
    # Check if we got a successful result
    if not result_queue.empty():
        return result_queue.get()
    
    # If thread is still running after timeout, check if it's retrying due to quota
    # Give it one more second to see if we get a quota error
    if thread.is_alive():
        thread.join(timeout=2)  # Wait 2 more seconds
        if not exception_queue.empty():
            e = exception_queue.get()
            error_str = str(e).lower()
            error_msg = str(e)
            if "quota" in error_str or "429" in error_msg or "resourceexhausted" in error_str:
                print(f"  → Model {model_name} exists but quota exceeded (model is valid): {error_msg[:100]}")
                return 'quota'
        
        # If still running, it's likely retrying - model probably not available
        print(f"  → Model {model_name} test timed out (model likely not available, retries taking too long)")
        return False
    
    # Should not reach here, but if we do, assume failure
    print(f"  → Model {model_name} test failed (unknown reason)")
    return False

def initialize_model(start_index=0):
    """Initialize chatModel by iterating through model candidates starting at start_index."""
    global chatModel, model_used, current_model_idx
    if start_index is None or start_index < 0:
        start_index = 0
    for idx in range(start_index, len(model_candidates)):
        canonical_name = model_candidates[idx]
        variant_names = expand_model_variants(canonical_name)
        for variant in variant_names:
            try:
                print(f"Attempting to initialize and test model: {variant}")
                
                # First test with google-genai SDK (faster, more reliable)
                test_result = test_model_with_google_genai(variant)
                
                if test_result is False:
                    print(f"✗ Model {variant} failed verification, trying next...\n")
                    continue
                elif test_result in (True, 'quota'):
                    if test_result == 'quota':
                        print(f"  → Model {variant} exists (quota issue, but will proceed)")
                    else:
                        print(f"  → Model {variant} verified with google-genai SDK")
                    
                    try:
                        chatModel = ChatGoogleGenerativeAI(model=variant, temperature=0.7)
                        model_used = variant
                        current_model_idx = idx
                        print(f"✓ Successfully initialized model: {variant}")
                        if test_result == 'quota':
                            print(f"  ⚠ Note: You may encounter quota limits with this model\n")
                        else:
                            print()
                        return True
                    except Exception as e:
                        print(f"  → Warning: Could not create langchain wrapper: {str(e)[:100]}")
                        print(f"✗ Model {variant} failed langchain initialization, trying next...\n")
                        continue
                else:
                    print(f"  → google-genai SDK not available, using langchain testing (slower)...")
                    test_model_instance = ChatGoogleGenerativeAI(model=variant, temperature=0.3)
                    test_result = test_model(test_model_instance, variant)
                    
                    if test_result in (True, 'quota'):
                        if test_result == 'quota':
                            print(f"  → Model {variant} exists (quota issue, but will proceed)")
                        chatModel = test_model_instance
                        model_used = variant
                        current_model_idx = idx
                        print(f"✓ Successfully initialized model: {variant}")
                        if test_result == 'quota':
                            print(f"  ⚠ Note: You may encounter quota limits with this model\n")
                        else:
                            print()
                        return True
                    else:
                        print(f"✗ Model {variant} failed verification, trying next...\n")
                        continue
            except Exception as e:
                print(f"✗ Failed to initialize model {variant}: {str(e)[:150]}\n")
                continue
    return False

# First, try to list available models to help with selection
print("Checking available models...")
available_models = list_available_models()
if available_models:
    print(f"  → Found {len(available_models)} available models")
    # Filter candidates to only include available models
    filtered_candidates = [m for m in model_candidates if any(avail in m or m in avail for avail in available_models)]
    if filtered_candidates:
        print(f"  → Filtered to {len(filtered_candidates)} potentially available models")
        model_candidates = filtered_candidates + [m for m in model_candidates if m not in filtered_candidates]
else:
    print("  → Could not list models, will try all candidates")

print()

if not initialize_model(0):
    raise ValueError(
        f"Failed to initialize and verify any Gemini model. Tried: {', '.join(model_candidates)}. "
        "Please check your GOOGLE_API_KEY and ensure it has access to Gemini models. "
        "You can also set GEMINI_MODEL in your .env file to specify a model name."
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = None
rag_chain = None

def rebuild_rag_chain():
    global question_answer_chain, rag_chain
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

rebuild_rag_chain()

def try_fallback_model(reason=""):
    """Switch to the next Gemini model if available and rebuild the RAG chain."""
    global current_model_idx
    next_index = (current_model_idx if current_model_idx is not None else -1) + 1
    if next_index >= len(model_candidates):
        print("No additional Gemini models available for fallback.")
        return False
    
    reason_text = f" due to {reason}" if reason else ""
    print(f"Attempting to switch to fallback Gemini model{reason_text}...")
    if initialize_model(next_index):
        rebuild_rag_chain()
        print(f"✓ Switched to fallback Gemini model: {model_used}\n")
        return True
    
    print("✗ Unable to switch to a fallback Gemini model.\n")
    return False


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if "msg" not in request.form:
            return "Error: No message provided", 400
        
        msg = request.form["msg"]
        if not msg or not msg.strip():
            return "Error: Message cannot be empty", 400
        
        print(f"\n{'='*60}")
        print(f"User Question: {msg}")
        print(f"Using model: {model_used}")
        print(f"{'='*60}")
        
        # Retry logic for rate limit errors
        max_retries = 3
        import time
        import re
        
        while True:
            retry_delay = 1  # Start with 1 second for each model
            fallback_triggered = False
            
            for attempt in range(max_retries):
                try:
                    # Invoke the RAG chain with the user's question
                    response = rag_chain.invoke({"input": msg})
                    answer = response.get("answer", "Sorry, I couldn't generate a response.")

                    sources_block = _build_sources_block(response.get("context"))
                    if sources_block:
                        answer = f"{answer.rstrip()}\n\n{sources_block}"
                    
                    # Log the retrieved context for debugging
                    if "context" in response:
                        context_length = len(str(response.get("context", "")))
                        print(f"Retrieved context length: {context_length} characters")
                    
                    # Ensure we have a meaningful answer
                    if not answer or answer.strip() == "" or len(answer.strip()) < 10:
                        answer = "I apologize, but I couldn't find sufficient information to answer your question. Please try rephrasing your question or ask about a different medical topic."
                    
                    print(f"Response length: {len(answer)} characters")
                    print(f"Response preview: {answer[:200]}...")
                    return str(answer)
                except Exception as model_error:
                    error_msg = str(model_error)
                    error_str_lower = error_msg.lower()
                    
                    # Check for rate limit / quota errors (429) - retry with backoff
                    if ("quota" in error_str_lower or "429" in error_msg or 
                        "resourceexhausted" in error_str_lower or "rate limit" in error_str_lower):
                        
                        if attempt < max_retries - 1:  # Not the last attempt
                            retry_time = retry_delay
                            if "retry in" in error_str_lower:
                                match = re.search(r'retry in ([\d.]+)s', error_str_lower)
                                if match:
                                    retry_time = min(float(match.group(1)), 60)  # Cap at 60 seconds
                            
                            print(f"Rate limit error (attempt {attempt + 1}/{max_retries}), retrying in {retry_time:.1f}s...")
                            time.sleep(retry_time)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            # Last attempt failed, try switching models before surfacing error
                            switched = try_fallback_model("Gemini rate limit/quota exhaustion")
                            if switched:
                                fallback_triggered = True
                                break  # Break out of retry loop and restart with the new model
                            
                            retry_time = None
                            if "retry in" in error_str_lower:
                                match = re.search(r'retry in ([\d.]+)s', error_str_lower)
                                if match:
                                    retry_time = float(match.group(1))
                            
                            user_message = (
                                "⚠️ Rate limit exceeded: You've hit the API quota limit. "
                                "Please wait a moment and try again."
                            )
                            if retry_time:
                                user_message += f" (Retry after {int(retry_time)} seconds)"
                            
                            print(f"Rate limit error after {max_retries} attempts - returning user-friendly message")
                            return user_message, 429
                    
                    # For other errors, handle them below (break out of retry loop)
                    # Re-raise to be caught by outer exception handler
                    raise model_error
            
            if fallback_triggered:
                print("Retrying user request with fallback Gemini model...\n")
                continue  # Restart while loop with new model
            
            # This should not be reached (loop should return or raise)
            return "Error: Unexpected error in retry loop", 500
        
    except Exception as model_error:
            error_msg = str(model_error)
            error_str_lower = error_msg.lower()
            print(f"Model error: {error_msg}")
            
            # Check if it's a model not found error
            if "not found" in error_str_lower or "404" in error_msg or "not supported" in error_str_lower:
                print(f"Model {model_used} failed during use, attempting to find working model...")
                return (
                    f"Error: The Gemini model '{model_used}' is not available with your API key. "
                    "Please restart the application or set GEMINI_MODEL=gemini-pro in your .env file.",
                    500
                )
            
            # Check for authentication errors
            elif "unauthorized" in error_str_lower or "401" in error_msg or "invalid api key" in error_str_lower:
                return (
                    "Error: Invalid API key. Please check your GOOGLE_API_KEY in the .env file.",
                    401
                )
            
            # Generic error - return user-friendly message
            else:
                # Don't expose internal error details to users
                print(f"Unexpected error: {error_msg}")
                return (
                    "Sorry, I encountered an error while processing your request. "
                    "Please try again in a moment.",
                    500
                )
                
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return (
            "Sorry, an unexpected error occurred. Please try again.",
            500
        )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)