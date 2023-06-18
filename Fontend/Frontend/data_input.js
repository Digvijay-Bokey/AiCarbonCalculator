var formSections = document.querySelectorAll('.form-section');
var currentSection = 0;
var predictButton = document.getElementById('predictButton');

formSections[currentSection].classList.add('show');

formSections.forEach(function(section, index) {
    var input = section.querySelector('input');

    input.addEventListener('input', function() {
        if (input.value !== '' && index === currentSection) {
            currentSection++;
            if (currentSection < formSections.length) {
                formSections[currentSection].classList.add('show');
                predictButton.style.marginTop = currentSection * 30 + 'px';
            } else {
                predictButton.disabled = false;
            }
        }
    });
});

document.getElementById('dataForm').addEventListener('submit', function (event) {
    event.preventDefault();
    window.location.href = 'results.html';
});
