document.getElementById('uploadButton').addEventListener('click', function() {
    var fileInput = document.getElementById('fileUpload');
    var file = fileInput.files[0];
    var reader = new FileReader();

    reader.onload = function(e) {
        var text = reader.result;
        document.getElementById('outputArea').innerText = text;
    }

    reader.readAsText(file);
});
