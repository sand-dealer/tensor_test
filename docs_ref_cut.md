## Testable docstrings

```
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.
  ...

  >>> t1 = [[1, 2, 3], [4, 5, 6]]
  >>> t2 = [[7, 8, 9], [10, 11, 12]]
  >>> concat([t1, t2], 0)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12]], dtype=int32)>

  <... more description or code snippets ...>

  Args:
    values: A list of `tf.Tensor` objects or a single `tf.Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

    Returns:
      A `tf.Tensor` resulting from concatenation of the input tensors.
  """
```

Note: TensorFlow DocTest uses TensorFlow 2 and Python 3.

### Make the code testable with DocTest

Currently, many docstrings use backticks (```) to identify code. To make the
code testable with DocTest:

*   Remove the backticks (```) and use the left-brackets (>>>) in front of each
    line. Use (...) in front of continued lines.
*   Add a newline to separate DocTest snippets from Markdown text to
    render properly on tensorflow.org.

### Docstring considerations

*   *Use symbols*: In DocTest you can directly access symbols defined in the
    same file. To use a symbol that’s not defined in the current file, please
    use TensorFlow’s public API `tf.xxx` instead of `xxx`. As you can see in the
    example below, <code>&#96;random.normal&#96;</code> is accessed via
    <code>&#96;tf.random.normal&#96;</code>. This is because
    <code>&#96;random.normal&#96;</code> is not visible in `NewLayer`.

    ```
    def NewLayer():
      “””This layer does cool stuff.

      Example usage:

      >>> x = tf.random.normal((1, 28, 28, 3))
      >>> new_layer = NewLayer(x)
      >>> new_layer
      <tf.Tensor: shape=(1, 14, 14, 3), dtype=int32, numpy=...>
      “””
    ```

*   *Non-deterministic output*: Use ellipsis(`...`) for the uncertain parts and
    DocTest will ignore that substring.

    ```
    >>> x = tf.random.normal((1,))
    >>> print(x)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=..., dtype=float32)>
    ```

*   *Multi-line blocks*: DocTest is strict about the difference between a single
    and a multi-line statement. Note the usage of (...) below:

    ```
    >>> if x > 0:
    ...   print("X is positive")
    >>> model.compile(
    ...   loss="mse",
    ...   optimizer="adam")
    ```

*   *Exceptions*: Exception details are ignored except the Exception that’s
    raised. See
    [this](https://docs.python.org/3/library/doctest.html#doctest.IGNORE_EXCEPTION_DETAIL)
    for more details.

    ```
    >>> np_var = np.array([1, 2])
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
    ```

### Test on your local machine

There are two ways to test the code in the docstring locally:

*   If you are only changing the docstring of a class/function/method, then you
    can test it by passing that file's path to
    [tf_doctest.py](https://www.tensorflow.org/code/tensorflow/tools/docs/tf_doctest.py).
    For example:

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">python tf_doctest.py --file=&lt;file_path&gt;</code>
    </pre>

    This will run it using your installed version of TensorFlow. To be sure
    you're running the same code that you're testing:

    *   Use an up to date [tf-nightly](https://pypi.org/project/tf-nightly/)
        `pip install -U tf-nightly`
    *   Rebase your pull request onto a recent pull from
        [TensorFlow's](https://github.com/tensorflow/tensorflow) master branch.

*   If you are changing the code and the docstring of a class/function/method,
    then you will need to
    [build TensorFlow from source](../../install/source.md). Once you are setup
    to build from source, you can run the tests:

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest</code>
    </pre>

    or

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest -- --module=ops.array_ops</code>
    </pre>

    The `--module` is relative to `tensorflow.python`.
