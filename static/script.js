function predict() {
    // Get input data from the form
    var inputValue = parseFloat(document.getElementById('inputValue').value);

    // Make an AJAX request to the Flask endpoint
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            // Display the result on the page
            document.getElementById('result').innerHTML = 'Predicted Montly receipts count: ' + JSON.parse(xhr.responseText).prediction;
        }
    };

    // Send the data as JSON
    xhr.send(JSON.stringify({ inputValue: inputValue }));
}

