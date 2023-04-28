from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./envit5_model")  
model = AutoModelForSeq2SeqLM.from_pretrained("./envit5_model")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def whisper_predict():
    input = request.params['text']
    # read input text
    inputs = [input]

    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids, max_length=512)

    return outputs[0]

if __name__ == '__main__':
    app.run(port=5000)