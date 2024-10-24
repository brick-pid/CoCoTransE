main() {
    try{
    let correct = ArrayList<Bool>(
            largestPrimeFactor(15) == 5,
            largestPrimeFactor(27) == 3,
            largestPrimeFactor(63) == 7,
            largestPrimeFactor(330) == 11,
            largestPrimeFactor(13195) == 29
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}