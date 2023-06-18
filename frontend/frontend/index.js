document.getElementById('getStarted').addEventListener('click', function () {
    window.location.href = 'data_input.html';
});

document.getElementById('dataForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Create data object from form input...
    var data = {
        input1: document.getElementById('input1').value,
        input2: document.getElementById('input2').value,
        // ... add more inputs here
    };

    // Make a POST request to the Flask API
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        // Redirect to results page and pass along the prediction data
        window.location.href = `results.html?prediction=${data.prediction}`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

