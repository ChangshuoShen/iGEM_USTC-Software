// 不用再做时间戳，单纯进行一个打招呼即可
now = new Date(), hour = now.getHours()
if (hour < 6) {
    var hello = "Good Early Morning";
} else if (hour < 12) {
    var hello = "Good Morning";
} else if (hour < 17) {
    var hello = "Good Afternoon";
} else if (hour < 22) { 
    var hello = "Good Evening";
} else {
    var hello = "Good Night";
}
hello += ", Welcome To USTC-Software's Homepage"
