const ratings = document.querySelectorAll('.rating')
const ratingsContainer = document.querySelector('.ratings-container')
const sendBtn = document.querySelector('#send')
const panel = document.querySelector('#panel')
const feedbackTextarea = document.querySelector('#feedback')
let selectedRating = 'Satisfied'

ratingsContainer.addEventListener('click', (e) => {
    if (e.target.closest('.rating')) {
        removeActive()
        const rating = e.target.closest('.rating')
        rating.classList.add('active')
        selectedRating = rating.querySelector('small').innerText
    }
})

function removeActive() {
    ratings.forEach(rating => rating.classList.remove('active'))
}
