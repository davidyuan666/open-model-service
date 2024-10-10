$(document).ready(function() {
    $('#generate').click(function() {
        var prompt = $('#prompt').val();
        if (prompt) {
            $.ajax({
                url: '/image/sd/generate',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ text: prompt }),
                beforeSend: function() {
                    $('#generate').prop('disabled', true).text('Generating...');
                },
                success: function(response) {
                    $('#generated-image').attr('src', 'data:image/png;base64,' + response.image).removeClass('hidden');
                    $('#generation-time').text('Generation time: ' + response.generation_time);
                },
                error: function(xhr, status, error) {
                    alert('Error generating image: ' + error);
                },
                complete: function() {
                    $('#generate').prop('disabled', false).text('Generate Image');
                }
            });
        } else {
            alert('Please enter a prompt');
        }
    });
});