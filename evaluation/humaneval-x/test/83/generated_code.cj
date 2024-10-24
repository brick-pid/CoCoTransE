main() {
    try{
    let correct = ArrayList<Bool>(
            startsOneEnds(1) == 1,
            startsOneEnds(2) == 18,
            startsOneEnds(3) == 180,
            startsOneEnds(4) == 1800,
            startsOneEnds(5) == 18000
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