---
layout: default
title: Oarph Oarphs
---

A place for Oarph to Oarph (sometimes about [`oarphpy`](https://github.com/pwais/oarphpy)).

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

