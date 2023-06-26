var formSections = document.querySelectorAll('.form-section');
var currentSection = 0;
var predictButton = document.getElementById('predictButton');

formSections[currentSection].classList.add('show');

formSections.forEach(function(section, index) {
    var select = section.querySelector('select');

    select.addEventListener('change', function() {
        if (select.value !== '' && index === currentSection) {
            currentSection++;
            if (currentSection < formSections.length) {
                formSections[currentSection].classList.add('show');
                formSections[currentSection].querySelector('select').disabled = false;
                predictButton.style.marginTop = currentSection * 1.5 + 'px';
            } else {
                predictButton.disabled = false;
            }
        }
    });
});

document.getElementById('dataForm').addEventListener('submit', function (event) {
    event.preventDefault();
    
    var data = new FormData();
    data.append('name', document.getElementById('input1').value);
    data.append('number', document.getElementById('number').value);
    
    fetch('/results', {
        method: 'POST',
        body: data,
    })
    .then(response => {
        window.location.href = `/results`;
    });  
});
