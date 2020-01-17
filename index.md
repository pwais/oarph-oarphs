---
layout: default
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

<img src="{{site.baseurl}}/assets/images/oarphoarph_alpha.png" width="100" height="100" />
