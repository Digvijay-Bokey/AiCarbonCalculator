window.onload = function() {
    var text = sessionStorage.getItem('textData');
    document.getElementById('outputArea').innerText = text;
};
