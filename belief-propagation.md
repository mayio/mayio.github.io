---
layout: page
title: Loopy Belief Propagation
subtitle: A five-post series, from message passing on a factor graph to a stereo matcher measured against ground truth
permalink: /belief-propagation/
---

These five posts are one argument in order. The first two build loopy belief
propagation on discrete variables and run it on a graph that actually has loops.
The third swaps the discrete distributions for Gaussians. The last two change the
semiring — max-sum in place of sum-product — and take the result out of the
textbook: first to data association for target tracking, then to sparse stereo
correspondence, where ground truth exists and the claim can be checked rather
than admired.

Read in order if the algorithm is new to you. Each post also stands alone, and
every one carries the Python it is describing.

{% assign series = site.data.series.belief_propagation %}
<ol class="series-list">
{% for item in series.posts %}
  {% assign post = site.posts | where: "path", item.path | first %}
  {% if post %}
  <li>
    <h3><a href="{{ post.url | relative_url }}">{{ post.title | strip_html }}</a></h3>
    <p>{{ item.blurb }}</p>
    <p class="series-date">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: site.date_format }}</time>
    </p>
  </li>
  {% endif %}
{% endfor %}
</ol>

## Where this goes next

The stereo work is ongoing: a sparse matcher running on an embedded platform,
where the interesting question is no longer whether max-sum finds the right
correspondences but what it costs per frame to find them. When there is
something measured worth reporting, it will appear here as a sixth part.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "CollectionPage",
  "@id": {{ page.url | absolute_url | jsonify }},
  "url": {{ page.url | absolute_url | jsonify }},
  "name": {{ page.title | strip_html | jsonify }},
  "description": {{ page.subtitle | strip_html | jsonify }},
  "author": {
    "@type": "Person",
    "@id": "https://www.mariolueder.com/aboutme/",
    "name": {{ site.author | jsonify }}
  },
  "hasPart": [
{%- assign emitted = 0 -%}
{%- for item in series.posts -%}
  {%- assign post = site.posts | where: "path", item.path | first -%}
  {%- if post -%}
    {%- if emitted > 0 -%},{%- endif %}
    {
      "@type": "BlogPosting",
      "@id": {{ post.url | absolute_url | jsonify }},
      "url": {{ post.url | absolute_url | jsonify }},
      "headline": {{ post.title | strip_html | jsonify }},
      "datePublished": {{ post.date | date_to_xmlschema | jsonify }}
    }
    {%- assign emitted = emitted | plus: 1 -%}
  {%- endif -%}
{%- endfor %}
  ]
}
</script>
