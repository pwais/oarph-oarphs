---
layout: default
---

<div style="max-width:80%">
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
    </li>
  {% endfor %}
  <li>a really really asdga asdg asdg asdg adsg dgas gad gsadsg adg adg adsds dgassdggs long line </li>
</ul>
</div>

<img src="{{site.baseurl}}/assets/images/oarphoarph_alpha.png" width="100" height="100" />
