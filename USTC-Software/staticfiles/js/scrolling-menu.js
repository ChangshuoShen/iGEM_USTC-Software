
let circ_eles = [];
let chapter_per = 1 / (chapter_array.length - 1);
const navline = document.getElementById("progress-line");

const locateCirc = () => {

    circ_eles = [];
    const chapter_ele = document.getElementById("chapter_items");
    chapter_ele.innerHTML = '';

    function generate_nav(id, ele, num) {
        const circ_ele = document.createElement('div');

        circ_ele.classList.add('chapter_item');
        circ_ele.classList.add('circ-passed');
        circ_ele.style.left = `${num * chapter_per * 100}%`;
        circ_ele.innerHTML = `<div class="circ"></div>
                              <div class="title"><a href="#${id}">${id.replace('-', ' ')}</a></div>`

        chapter_ele.appendChild(circ_ele);

        return [ele.getBoundingClientRect().top + document.documentElement.scrollTop, circ_ele];
    }

    let circ_num = 0;
    chapter_array.forEach(chapter_id => {
        const nodeEle = document.getElementById(chapter_id);
        const [ height, titleEle ] = generate_nav(chapter_id, nodeEle, circ_num);
        circ_eles.push({
          nodeEle,
          titleEle,
          height,
        });
        circ_num++;
      })

};

const DOMLoadedHandler = () => {
    locateCirc();
};

function debounce(fn,delay=5){
    let timer = null;
    return function() {
        if(timer) {
            clearTimeout(timer);
            timer = setTimeout(fn,delay);
        }
        else {
        timer = setTimeout(fn,delay);
        }
    }
}

const ScrollHandler = () => {

    const top = document.documentElement.scrollTop || document.body.scrollTop || window.pageYOffset;

    let pre_ele = null;
    let width = 0;
    let found = false;    // 标识是否已找到第一个经过章节

    for (let i = circ_eles.length - 1; i >= 0; i--) {
        const ele = circ_eles[i];
        ele.titleEle.classList.remove('circ-passed', 'circ-inactive', 'circ-active');

        if (!found) {
            if (top <= ele.height - 5) {
                ele.titleEle.classList.add('circ-inactive');
                pre_ele = ele;
            }
            else {
                if (pre_ele) {
                    width = i * chapter_per * 100;
                    width += (top - ele.height) / (pre_ele.height - ele.height) * chapter_per * 100;
                    found = true;
                    ele.titleEle.classList.add('circ-active');
                }
                else {
                    width = 100;
                    found = true;
                    ele.titleEle.classList.add('circ-active');
                }
            }
        }
        else {
            ele.titleEle.classList.add('circ-passed');
        }
    }

    navline.style.width = `${Math.min(width, 100)}%`;

};

window.addEventListener('DOMContentLoaded', DOMLoadedHandler);  // 当页面内容加载时，定位滚动菜单
window.addEventListener('resize', locateCirc);                  // 当页面大小改变时，重定位滚动菜单
window.addEventListener('scroll', debounce(ScrollHandler));     // 滚动时，调节progress条，显示滚动进度