이 시점에서 사용자는 다음 HTML이 생성됩니다.를 사용하여 주석을 추가할 수 있습니다.

<pre>  [example attr1="attr_content" attr2="attr_content"]
  {% raw %}<p>Bob is an excellent builder!</p>
  {% endraw %}[/example]
  <p>Text</p>
</pre>

이것은 데모이기 때문에 댓글을 지속하지 않으므로 한 번에 하나의 댓글만 추가할 수 있습니다.

<code>
  <p>Text</p>
  [example attr1="attr_content" attr2="attr_content"]
  {% raw %}<p>Bob is an excellent builder!</p>
  {% endraw %}[/example]
</code>

<code>
  <p>Text</p>
  [sourcecode][/sourcecode]
  <p>Text</p>
</code>

<pre>  [sourcecode][/sourcecode]
  <p>Text</p>
  [sourcecode][/sourcecode]
</pre>
