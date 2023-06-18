var formSections = Array.from(document.querySelectorAll('.form-section'));
var currentSection = 0;
var predictButton = document.getElementById('predictButton');

// Add 'show' class to the first section
formSections[currentSection].classList.add('show');

formSections.forEach(function(section, index) {
    var input = section.querySelector('input');

    input.addEventListener('input', function() {
        if (input.value !== '' && index === currentSection) {
            currentSection++;
            if (currentSection < formSections.length) {
                formSections[currentSection].classList.add('show');
            } else {
                predictButton.disabled = false; // Enable the "Predict" button
            }
        }
    });
});

document.getElementById('dataForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Handle the data input and submission here...

    window.location.href = 'results.html';
});
