
<script type="application/json">
  {
    "ad-attributes": {
      // ad server configuration
    }
  }
</script>

* An AMP web page

<script type="application/json">
  {
    "ad-attributes": {
      "type": "doubleclick",
      "data-slot": "/30497360/a4a/amp_story_dfp_example"
    }
  }
</script>

* A non-AMP web page

<script type="application/json">
 {
   "ad-attributes": {
     "type": "doubleclick",
     "data-slot": "/30497360/a4a/amp_story_dfp_example",
     "additional_data": "additional_data_information"
   }
 }
</script>

* A some web page

[sourcecode:html]
<amp-story>
  <amp-story-auto-ads>
    <script type="application/json">
      {
        "ad-attributes": {
          //ad server configuration
        }
      }
    </script>
  </amp-story-auto-ads>
  <amp-story-page>
  ...
</amp-story>
[/sourcecode]

* A next generation of web pages

[sourcecode:html]
<amp-story>
  <amp-story-auto-ads>
    <script type="application/json">
     {
       "ad-attributes": {
         "type": "doubleclick",
         "data-slot": "/30497360/a4a/amp_story_dfp_example",
         "additional_data": "additional_data_information"
       }
     }
    </script>
  </amp-story-auto-ads>
  <amp-story-page>
  ...
</amp-story>
[/sourcecode]
