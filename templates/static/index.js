document.addEventListener('DOMContentLoaded', function() {
  let form = document.getElementById('dataForm');

  form.addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData(form);
    let data = Object.fromEntries(formData);

    fetch('/predict', {  // Changed the URL here
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    }).then(response => response.json())
      .then(data => console.log(data));
  });
});
