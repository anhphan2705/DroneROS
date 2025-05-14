console.log('[HMI] main.js loaded');

document.addEventListener('DOMContentLoaded', () => {
  console.log('[HMI] DOMContentLoaded');

  const sel        = document.getElementById('topic-select');
  const video      = document.getElementById('video-feed');
  const overlay    = document.getElementById('video-overlay');
  const launchBtn  = document.getElementById('launch-btn');
  const stopBtn    = document.getElementById('stop-btn');
  const statusText = document.getElementById('status-text');
  const STALE_SEC  = 2;

  if (!sel || !video || !overlay || !launchBtn || !stopBtn || !statusText) {
    console.error('[HMI] missing one or more required elements');
    return;
  }

  //
  // TOPIC + LIVENESS LOGIC
  //
  const updateTopics = async () => {
    let topics = [];
    try {
      const res  = await fetch('/api/topics');
      topics     = (await res.json()).topics || [];
    } catch (e) {
      console.error('[HMI] failed to fetch topics', e);
    }

    const prev = sel.value;
    sel.innerHTML = '';
    if (!topics.length) {
      sel.add(new Option('No feeds available','',false,false));
      return;
    }
    topics.forEach(t => sel.add(new Option(t,t)));
    sel.value = topics.includes(prev) ? prev : topics[0];
    onTopicChange();
  };

  const onTopicChange = () => {
    const topic = sel.value;
    console.log('[HMI] switching to', topic);
    video.src = topic
      ? `/video_feed?topic=${encodeURIComponent(topic)}`
      : '';
    checkLiveness();
  };

  const checkLiveness = async () => {
    const topic = sel.value;
    if (!topic) {
      overlay.style.display = 'flex';
      return;
    }
    try {
      const res        = await fetch(`/api/last_frame?topic=${encodeURIComponent(topic)}`);
      const { last_frame } = await res.json();
      const age        = (Date.now()/1000) - last_frame;
      overlay.style.display = age > STALE_SEC ? 'flex' : 'none';
    } catch (e) {
      overlay.style.display = 'flex';
    }
  };

  sel.addEventListener('change', onTopicChange);

  //
  // LAUNCH/STOP + STATUS LOGIC
  //
  const updateStatus = async () => {
    try {
      const res  = await fetch('/api/status');
      const data = await res.json();
      const running = Object.values(data).some(v => v === true);
      statusText.textContent = running ? 'Running' : 'Idle';
    } catch (e) {
      statusText.textContent = 'Error';
      console.error('[HMI] status fetch error:', e);
    }
  };

  const sendCommand = async (endpoint) => {
    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name:'mission'})
      });
      updateStatus();
    } catch (e) {
      console.error('[HMI] command error:', e);
    }
  };

  launchBtn.addEventListener('click', () => sendCommand('/api/launch'));
  stopBtn.addEventListener('click', () => sendCommand('/api/stop'));

  //
  // INITIALIZE
  //
  updateTopics();
  setInterval(updateTopics, 5000);

  setInterval(checkLiveness, 1000);

  updateStatus();
  setInterval(updateStatus, 2000);
});