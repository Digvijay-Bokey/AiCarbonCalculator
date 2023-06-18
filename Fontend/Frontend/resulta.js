window.onload = function() {
    var userInput = sessionStorage.getItem('userInput');
    // Use the user's input to generate the results
    var results = 'Your input was: ' + userInput + '\nYour carbon impact prediction will go here.';
    document.getElementById('results').innerText = results;
};
