main() {
    try{
    var map1 = HashMap<String, Int64>();
    map1.put("a", 2);
    map1.put("b", 2);
    var map2 = HashMap<String, Int64>();
    map2.put("a", 2);
    map2.put("b", 2);
    var map3 = HashMap<String, Int64>();
    map3.put("a", 1);
    map3.put("b", 1);
    map3.put("c", 1);
    map3.put("d", 1);
    map3.put("g", 1);
    var map4 = HashMap<String, Int64>();
    map4.put("r", 1);
    map4.put("t", 1);
    map4.put("g", 1);
    var map5 = HashMap<String, Int64>();
    map5.put("b", 4);
    var map6 = HashMap<String, Int64>();
    map6.put("r", 1);
    map6.put("t", 1);
    map6.put("g", 1);
    var map7 = HashMap<String, Int64>();
    var map8 = HashMap<String, Int64>();
    map8.put("a", 1);
    let correct = ArrayList<Bool>(
            histogram("a b b a") == map1,
            histogram("a b c a b") == map2,
            histogram("a b c d g") == map3,
            histogram("r t g") == map4,
            histogram("b b b b a") == map5,
            histogram("r t g") == map6,
            histogram("") == map7,
            histogram("a") == map8
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