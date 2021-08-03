At this point, the user can add a comment using the
The following HTML is generated:

<pre>
  [example attr1="attr_content" attr2="attr_content"]
  {% raw %}<p>Bob is an excellent builder!</p>
  {% endraw %}[/example]
  <p>Text</p>
</pre>

Because this is a demo, we are not persisting comments, so it’s only possible to add one comment at the time

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

<pre>
  [sourcecode][/sourcecode]
  <p>Text</p>
  [sourcecode][/sourcecode]
</pre>
