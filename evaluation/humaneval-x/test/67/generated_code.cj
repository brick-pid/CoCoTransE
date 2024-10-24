main() {
    try{
    let correct = ArrayList<Bool>(
            fruitDistribution("5 apples and 6 oranges",19) == 8,
            fruitDistribution("5 apples and 6 oranges",21) == 10,
            fruitDistribution("0 apples and 1 oranges",3) == 2,
            fruitDistribution("1 apples and 0 oranges",3) == 2,
            fruitDistribution("2 apples and 3 oranges",100) == 95,
            fruitDistribution("2 apples and 3 oranges",5) == 0,
            fruitDistribution("1 apples and 100 oranges",120) == 19
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