main() {
    try{
    let correct = ArrayList<Bool>(
            chooseNum(12, 15) == 14,
            chooseNum(13, 12) == -1,
            chooseNum(33, 12354) == 12354,
            chooseNum(5234, 5233) == -1,
            chooseNum(6, 29) == 28,
            chooseNum(27, 10) == -1,
            chooseNum(7, 7) == -1,
            chooseNum(546, 546) == 546
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