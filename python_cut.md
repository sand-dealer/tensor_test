## Install TensorFlow Lite for Python

If you're running Debian Linux or a derivative of Debian (including Raspberry Pi
OS), you should install from our Debian package repo. This requires that you add
a new repo list and key to your system and then install as follows:

<pre class="devsite-terminal">
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
<code class="devsite-terminal"
>curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
</code><code class="devsite-terminal"
>sudo apt-get update
</code><code class="devsite-terminal"
>sudo apt-get install python3-tflite-runtime</code>
</pre>

For all other systems, you can install with pip:

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

## Run an inference using tflite_runtime

Instead of importing `Interpreter` from the `tensorflow` module, you now need to
import it from `tflite_runtime`.

```python
import tensorflow as tf
```

So it instead reads:

```python
import tflite_runtime.interpreter as tflite
```

And then change this line:

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

So it reads:

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```
