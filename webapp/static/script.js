// Global variable to store the selected file (via input or drag & drop)
let selectedFile = null;

// Function to handle file selection via input or drop
function handleFileSelect(event) {
  const file = event.target.files ? event.target.files[0] : event.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    previewImageFromFile(file);

    // Hide drag & drop area and Browse button
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('browseButton').style.display = 'none';
    document.getElementById('removeButton').style.display = 'inline-block';
  }
}

// Function to preview the image
function previewImageFromFile(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = document.getElementById('previewImage');
    img.src = e.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// Drag & drop events
function allowDrag(event) {
  event.preventDefault();
}

function handleDrop(event) {
  event.preventDefault();
  handleFileSelect(event);
}

// Add drag & drop events to the corresponding area
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', allowDrag, false);
dropZone.addEventListener('drop', handleDrop, false);

// Function to send the image to the backend and display the diagnosis
function submitImage() {
  if (!selectedFile) {
    alert('Please select an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);

  // Show loading message
  document.getElementById('diagnosisResult').innerText = 'Processing...';

  // Start the fetch request and chain the .then method
  fetch('/diagnose', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById('diagnosisResult').innerText = data.diagnosis;
      // Show feedback prompt to the user
      showFeedback();
    })
    .catch(error => {
      console.error('Error:', error);
      document.getElementById('diagnosisResult').innerText = 'Error processing image.';
    });
}


// Function to remove the image and restore initial state
function removeImage() {
  selectedFile = null;
  // Hide preview image
  const img = document.getElementById('previewImage');
  img.src = '';
  img.style.display = 'none';
  // Restore drag & drop area and Browse button
  document.getElementById('dropZone').style.display = 'flex';
  document.getElementById('browseButton').style.display = 'inline-block';
  // Hide Remove button
  document.getElementById('removeButton').style.display = 'none';
  // Clear file input
  document.getElementById('imageInput').value = "";
  // Restore result text
  document.getElementById('diagnosisResult').innerText = 'No diagnosis yet.';
}

// Function to fetch model information and update the footer
function fetchModelInfo() {
  fetch('/model-info')
    .then(response => response.json())
    .then(data => {
      const modelInfo = document.getElementById('modelInfo');
      modelInfo.innerHTML = `<strong>Model: ${data.name} - Version: ${data.version}</strong>`;
    })
    .catch(error => {
      console.error('Error fetching model info:', error);
    });
}

function sendFeedback(feedbackValue) {
  // Optionally, you could generate or attach an image_id if you need to correlate feedback
  const payload = {
    image_id: "some_image_identifier", // you may want to generate this or capture from context
    feedback: feedbackValue
  };
  
  fetch('/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => {
    console.log("Feedback logged:", data);
    // Hide feedback container once submitted
    document.getElementById('feedbackContainer').style.display = 'none';
  })
  .catch(error => {
    console.error('Error sending feedback:', error);
  });
}

// Call this function after displaying the diagnosis result:
function showFeedback() {
  document.getElementById('feedbackContainer').style.display = 'block';
}

// In your submitImage() function, once you have the diagnosis result, you may call:



// Call model info function on page load
window.onload = fetchModelInfo;

