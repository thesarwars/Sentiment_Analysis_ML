document.addEventListener('DOMContentLoaded', function(){
    var form = document.getElementById('predSentiment');
    form.addEventListener('submit', function (e){
        e.preventDefault();
        var formData = new FormData(form);

        fetch('127.0.0.1:8000',{
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result-sentiment').textContent=data.sentiment;
            document.getElementById('result-text').textContent=data.text
        })
        .catch(error => console.error('Error', error))
    })
})