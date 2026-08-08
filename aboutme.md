---
layout: page
title: About me
subtitle: 'Software engineer in Hamburg, working on object tracking from lidar point clouds and on probabilistic inference on factor graphs.'
---

Hello there! 

I'm Mario, a software engineer working on Object Tracking — the core intelligence behind autonomous driving — in Hamburg, Germany.

I created this space to explore the algorithms and concepts that define my field. I'm keen to fill the knowledge gaps I've encountered myself and provide truly in-depth explanations. Feel free to explore both this blog and my GitHub repository for practical examples.

## My Focus: From Point Cloud to Real-Time Kinematics

Accurate environmental sensing is non-negotiable for advanced driver-assistance and autonomous systems. While vehicles rely on multiple sensors, my work concentrates on processing data from laser scanners (LiDAR), which generate massive point clouds.

My challenge is developing algorithms to extract objects — such as cars, trucks, pedestrians, and small obstacles — from this data in real-time. This involves precisely estimating their pose, shape, and kinematics to enable safe autonomous navigation.

My core expertise includes:

* Multiple Extended Target Tracking (METT)
* Shape Estimation
* Clustering

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "@id": "https://www.mariolueder.com/aboutme.html",
  "url": "https://www.mariolueder.com/aboutme.html",
  "name": "Mario Lüder",
  "jobTitle": "Software Engineer",
  "description": "Software engineer working on object tracking from lidar point clouds for autonomous driving, and on probabilistic inference on factor graphs.",
  "knowsAbout": [
    "Loopy Belief Propagation",
    "Factor Graphs",
    "Data Association",
    "Multiple Extended Target Tracking",
    "Stereo Vision",
    "Lidar Point Cloud Processing"
  ],
  "homeLocation": {
    "@type": "Place",
    "name": "Hamburg, Germany"
  },
  "sameAs": [
    "https://github.com/mayio",
    "https://www.linkedin.com/in/mariolueder",
    "https://www.injournal.de"
  ]
}
</script>

<!--
  sameAs is the main lever for getting Google to treat "Mario Lüder" as an
  entity rather than a string, so add each of these to the array above as soon
  as it exists:
    "https://arxiv.org/a/<arxiv-author-id>",
    "https://orcid.org/<orcid>",
    "https://scholar.google.com/citations?user=<id>"

  A "worksFor" property belongs here too, but the employer is named nowhere in
  this repository and is not mine to fill in:
    "worksFor": { "@type": "Organization", "name": "...", "url": "..." },
-->

