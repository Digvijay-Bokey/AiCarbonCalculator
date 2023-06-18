document.getElementById('submitButton').addEventListener('click', function(event) {
    event.preventDefault();
    // Get the user's input and store it
    var exampleInput = document.getElementById('exampleInput').value;
    sessionStorage.setItem('userInput', exampleInput);
    window.location.href = 'results.html';
});
