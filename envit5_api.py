from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./envit5_model")  
model = AutoModelForSeq2SeqLM.from_pretrained("./envit5_model")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def whisper_predict():
    # read translation mode
    mode = request.args.get('mode')

    # read input text
    input = request.data.decode('utf-8')

    if mode == "en2vi":
        input = "en: " + input
    else:
        input = "vi: " + input
    inputs = [input]

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)

    outputs_tokens = model.generate(tokenized_inputs.input_ids, max_length=4096)

    outputs = tokenizer.batch_decode(outputs_tokens, skip_special_tokens=True)

    output = outputs[0][4:]

    return output

if __name__ == '__main__':
    app.run(port=5000)