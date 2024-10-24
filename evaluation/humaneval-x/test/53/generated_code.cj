main() {
    try{
    let rand = Random(42);
    let correct = ArrayList<Bool>(
            add(0, 1) == 1,
            add(1, 0) == 1,
            add(2, 3) == 5,
            add(5, 7) == 12,
            add(7, 5) == 12
    );
    if (correct.contains(false)) {
        print(correct)
    }
    for (i in 0..100) {
        var x : Int64 = rand.nextInt64(1000);
        var y : Int64 = rand.nextInt64(1000);
        if (add(x, y) != x + y) {
            print("false")
        }
    }
        }
        catch(e: Exception)
    {
        print("false")
    }
}