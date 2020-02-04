---
layout: default
---

A place for Oarph to Oarph.

## Posts

<div style="max-width:80%">
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
</div>

