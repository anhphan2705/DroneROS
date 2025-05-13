// Polls /api/status every 2 seconds and updates status text
async function updateStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const running = Object.values(data).some(v => v === true);
      document.getElementById('status-text').textContent = running ? 'Running' : 'Idle';
    } catch (err) {
      document.getElementById('status-text').textContent = 'Error';
    }
  }
  
  // Send POST to /api/launch or /api/stop
  async function sendCommand(endpoint) {
    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'mission' })
      });
      updateStatus();
    } catch (err) {
      console.error(err);
    }
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    // Wire up buttons
    document.getElementById('launch-btn')
      .addEventListener('click', () => sendCommand('/api/launch'));
    document.getElementById('stop-btn')
      .addEventListener('click', () => sendCommand('/api/stop'));
  
    // Initial status and polling
    updateStatus();
    setInterval(updateStatus, 2000);
  });// Polls /api/status every 2 seconds and updates status text
  async function updateStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const running = Object.values(data).some(v => v === true);
      document.getElementById('status-text').textContent = running ? 'Running' : 'Idle';
    } catch (err) {
      document.getElementById('status-text').textContent = 'Error';
    }
  }
  
  // Send POST to /api/launch or /api/stop
  async function sendCommand(endpoint) {
    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'mission' })
      });
      updateStatus();
    } catch (err) {
      console.error(err);
    }
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    // Wire up buttons
    document.getElementById('launch-btn')
      .addEventListener('click', () => sendCommand('/api/launch'));
    document.getElementById('stop-btn')
      .addEventListener('click', () => sendCommand('/api/stop'));
  
    // Initial status and polling
    updateStatus();
    setInterval(updateStatus, 2000);
  });