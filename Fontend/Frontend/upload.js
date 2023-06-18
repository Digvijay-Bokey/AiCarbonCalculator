document.getElementById('uploadButton').addEventListener('click', function() {
    var fileInput = document.getElementById('fileUpload');
    var file = fileInput.files[0];
    var reader = new FileReader();

    reader.onload = function(e) {
        var text = reader.result;
        sessionStorage.setItem('textData', text);
        window.location.href = 'output.html';
    }

    reader.readAsText(file);
});
