from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

from helper import build_helper
from model import Model 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

def worker(lipsum, helper):
    lipsum.sess.run(helper.initializer)
    while True:
        try:
            outputs, _ = lipsum.sess.run([lipsum.outputs, lipsum.train_op])
            socketio.emit('lipsum', {'text': ''.join(helper.ix_to_char[i] for i in outputs.sample_id[0])})
        except tf.errors.OutOfRangeError:
            lipsum.sess.run(helper.initializer)


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

if __name__ == "__main__":
    helper = build_helper()
    lipsum = Model(helper)
    t = threading.Thread(target=worker, args=(lipsum, helper))
    t.start()

    socketio.run(app)