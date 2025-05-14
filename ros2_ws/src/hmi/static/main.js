// static/main.js

console.log('[HMI] main.js loaded');

document.addEventListener('DOMContentLoaded', () => {
  console.log('[HMI] DOMContentLoaded');

  const sel   = document.getElementById('topic-select');
  const video = document.getElementById('video-feed');

  if (!sel || !video) {
    console.error('[HMI] Missing #topic-select or #video-feed!');
    return;
  }

  // fetch & refresh topics, preserving the current selection
  const updateTopics = async () => {
    console.log('[HMI] fetching /api/topics…');
    let topics = [];

    try {
      const res  = await fetch('/api/topics');
      const body = await res.json();
      topics     = body.topics || [];
      console.log('[HMI] got topics:', topics);
    } catch (err) {
      console.error('[HMI] failed to fetch /api/topics:', err);
      return;
    }

    // 1) remember the old selection
    const prev = sel.value;

    // 2) rebuild the dropdown
    sel.innerHTML = '';
    if (topics.length === 0) {
      sel.add(new Option('No feeds available', '', false, false));
      video.src = '';
      return;
    }
    topics.forEach(t => sel.add(new Option(t, t)));

    // 3) restore old selection if still present, else pick the first
    const chosen = topics.includes(prev) ? prev : topics[0];
    sel.value = chosen;

    // 4) immediately update the video
    console.log('[HMI] setting topic to', chosen);
    video.src = `/video_feed?topic=${encodeURIComponent(chosen)}`;
  };

  // when the user manually picks a new topic
  sel.addEventListener('change', () => {
    const topic = sel.value;
    console.log('[HMI] selected topic:', topic);
    video.src = topic
      ? `/video_feed?topic=${encodeURIComponent(topic)}`
      : '';
  });

  // initial load + periodic refresh
  updateTopics();
  setInterval(updateTopics, 5000);
});