// static/main.js

// Polls /api/status every 2 seconds and updates the status badge
async function updateStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const running = Object.values(data).some(v => v === true);
    document.getElementById('status-text').textContent = running ? 'Running' : 'Idle';
  } catch (err) {
    document.getElementById('status-text').textContent = 'Error';
    console.error('Status fetch error:', err);
  }
}

// Sends a POST to /api/launch or /api/stop
async function sendCommand(endpoint) {
  try {
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'mission' })
    });
    updateStatus();
  } catch (err) {
    console.error('Command error:', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('topic-select');
  const video = document.getElementById('video-feed');

  // Fetch and render topic options every 5 seconds
  async function updateTopics() {
    try {
      const { topics } = await fetch('/api/topics').then(r => r.json());
      // clear existing options
      sel.innerHTML = '';
      // if no topics
      if (topics.length === 0) {
        const opt = document.createElement('option');
        opt.disabled = true;
        opt.textContent = 'No feeds available';
        sel.appendChild(opt);
        video.src = '';
        return;
      }
      // populate select
      topics.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        sel.appendChild(opt);
      });
      // auto-select first
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    } catch (e) {
      console.error('Failed to fetch topics:', e);
    }
  }

  // Initial topics fetch & polling
  updateTopics();
  setInterval(updateTopics, 5000);

  // When you pick a topic, swap the feed URL
  sel.addEventListener('change', () => {
    const topic = sel.value;
    if (topic) {
      video.src = `/video_feed?topic=${encodeURIComponent(topic)}`;
    }
  });

  // Wire buttons
  document.getElementById('launch-btn')
    .addEventListener('click', () => sendCommand('/api/launch'));
  document.getElementById('stop-btn')
    .addEventListener('click', () => sendCommand('/api/stop'));

  // Start status polling
  updateStatus();
  setInterval(updateStatus, 2000);
});