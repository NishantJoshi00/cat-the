#![cfg_attr(
    feature = "nightly-features",
    feature(unboxed_closures, fn_traits, tuple_trait)
)]

mod chapter_1 {

    pub fn compose<A, B, C, F1, F2>(f1: F1, f2: F2) -> impl Fn(A) -> C
    where
        F1: Fn(A) -> B,
        F2: Fn(B) -> C,
    {
        move |a| f2(f1(a))
    }
    fn identity<T>(item: T) -> T {
        item
    }

    pub fn main() {
        let add_1 = |x: i32| x + 1;
        let add_2 = |x: i32| x + 2;
        let add_3 = compose(add_1, add_2);
        dbg!(add_3(12) == 15);
        let just_add_2 = compose(add_2, identity);
        dbg!(just_add_2(12) == 14);
    }
}

mod chapter_2 {
    use std::{collections::HashMap, hash::Hash};
    
    #[cfg(not(feature = "nightly-features"))]
    pub fn main() {
        dbg!(fact(2), 2);

        let mut new_fact = HOF::new(fact);

        dbg!(new_fact.call(2), 2);
    }

    #[cfg(feature = "nightly-features")]
    pub fn main() {
        let _new_fact = HOF::new(fact);
    }

    fn fact(n: i32) -> i32 {
        (1..=n).product()
    }

    /*
     *
     * 1. Functions with same input and same output are called as pure Functions
     * 2. Functions that can be implemented with the same formula for any type are called
     *    parametrically polymorphic.
     * 3. Functions to Bool are called predicates.
     *
     */

    #[cfg(not(feature = "nightly-features"))]
    struct HOF<U, V, F> 
    where
        F: Fn(U) -> V,
        U: Eq + Hash,
        V: Clone
    {
        func: F,
        memory: HashMap<U, V>
    }

    #[cfg(not(feature = "nightly-features"))]
    impl<U: Eq + Hash + Clone, V: Clone, F: Fn(U) -> V> HOF<U, V, F> {
        fn new(func: F) -> Self {
            Self {
                func,
                memory: Default::default()
            }
        }

        fn call(&mut self, args: U) -> V {
            if let Some(value) = self.memory.get(&args) {
                value.clone()
            } else {
                let output = (self.func)(args.clone());
                self.memory.insert(args, output.clone());
                output
            }
        }
    }


    #[cfg(feature = "nightly-features")]
    struct HOF<U, V, F>
    where
        F: Fn<U> + FnOnce<U, Output = V>,
        U: Eq + Hash + Clone + std::marker::Tuple,
        V: Clone,
    {
        func: F,
        memory: HashMap<U, V>,
    }

    #[cfg(feature = "nightly-features")]
    impl<U, V, F> HOF<U, V, F>
    where
        F: Fn<U> + FnOnce<U, Output = V>,
        U: Eq + Hash + Clone + std::marker::Tuple,
        V: Clone,
    {
        fn new(func: F) -> Self {
            Self {
                func,
                memory: HashMap::new(),
            }
        }
    }

    #[cfg(feature = "nightly-features")]
    impl<U: Eq + Hash + Clone + std::marker::Tuple, V: Clone, F: Fn<U> + FnOnce<U, Output = V>>
        FnMut<U> for HOF<U, V, F>
    {
        extern "rust-call" fn call_mut(&mut self, args: U) -> Self::Output {
            if let Some(value) = self.memory.get(&args) {
                value.clone()
            } else {
                let output = (self.func)(args.clone());
                self.memory.insert(args, output).unwrap()
            }
        }
    }

    #[cfg(feature = "nightly-features")]
    impl<U: Eq + Hash + Clone + std::marker::Tuple, V: Clone, F: Fn<U> + FnOnce<U, Output = V>>
        FnOnce<U> for HOF<U, V, F>
    {
        type Output = V;
        extern "rust-call" fn call_once(self, args: U) -> Self::Output {
            (self.func)(args)
        }
    }
}


mod chapter_3 {
    use std::marker::PhantomData;

    /*
     *
     * 1. There is a category of no objects
     * 2. Monoid is an embarrassingly simple but amazingly powerful concept. 
     * Traditionally, a monoid is defined as a set with a binary operation. All that's required
     * from this operation is that it's associative, and that there is one special element that
     * behaves like a unit with respect to it.
     *
     *
     */
    pub fn main() {}

    #[allow(dead_code)]
    struct Monoid_<T, F: Fn(T, T) -> T> {
        type_: PhantomData<T>,
        bin_op: F, // associative
        unit: T // bin_op(x, unit) -> x
    }

    #[allow(dead_code)]
    trait Monoid<T> {
        fn mempty() -> T;
        fn mappend(a: T, b: T) -> T;
    }
}

#[allow(dead_code)]
mod chapter_4 {
    fn main() {}

    type Writer<T> = (T, String);

    trait Kleisli<T, U, F> {
        type FOutput<S>;

        fn return_(item: F) -> Self;

        fn kleisli_op<'a>(f1: impl Fn(T) -> Self::FOutput<U> + 'a, f2: impl Fn(U) -> Self::FOutput<F> + 'a) -> Box<dyn Fn(T) -> Self::FOutput<F> + 'a>;
    }

    impl<T, U, F> Kleisli<T, U, F> for Writer<F> {
        type FOutput<S> = Writer<S>;
        fn return_(item: F) -> Self {
            (item, String::new())
        }

        fn kleisli_op<'a>(f1: impl Fn(T) -> Writer<U> + 'a, f2: impl Fn(U) -> Self + 'a) -> Box<dyn Fn(T) -> Self + 'a> {

            Box::new(move |x| {
                let a = f1(x);
                let b = f2(a.0);
                (b.0, a.1 + &b.1)
            })
        }
    }

}

#[allow(dead_code)]
mod chapter_5 {
    fn main() {}

    /* 
     * - The **initial object** is the object that has one and only one morphism going to any object
     * in the category.
     * - The **terminal object** is the object with one and only one morphism coming to it from any
     * object in the category.
     *
     * **initial object** ~ **co-terminal object**
     * **terinal object** ~ **co-initial object**
     *
     * Only difference being the direction of the morphism, and thus a dual is possible
     *
     * ---
     * Isomorphism is an invertible morphism; or a pair of morphisms, one being the inverse of the
     * other.
     *
     *
     * ### product
     * A product of two objects in any category using the same universal construction. Such a
     * product doesn't always exist, but when it does, it is unique up to a unique iso-morphism.
     *
     * A **product** of two objects a and b is the object c equipped with two projections such that
     * for any other object c' equipped with two projections there is a unique morphism m from c'
     * to c that factorizes those projections.
     *
     * example, being pair or (T, U)
     *
     * A **coproduct** of two objects a and b is the object c equipped with two injections such
     * that for any other object c' equipped with injections there is a unique morphism m from c to
     * c' that factorizes those injections.
     *
     * example, bool 
     *
     */
}


#[allow(dead_code)]
mod chapter_6 {
    pub fn main() {}

    /* 
     * The product of 2 types in a tuple of 2 elements <T, U>(T, U)
     */

    // Record example
    // compare tuple type vs struct type
    struct Element_(String, String, usize);
    
    // A proper record looks like:
    struct Element {
        name: String,
        symbol: String,
        atomic_number: usize
    }

    // Example of sum types is a Result<T, U>, where either or is the type of the data

    /*
     * Taken separately, product and sum types can be used to define a variety of useful data
     * structures, but the real strength comes from combining the two. Once again we are invoking
     * the power of composition.
     */
    /* 
     * Relationship between natural numbers and types
     * |    Number  |   Types           |
     * |    ---     |   ---             |
     * |    0       |   Void            |
     * |    1       |   ()              |
     * |    a + b   |   Result<a, b>    |
     * |    a x b   |   (a, b)          |
     * |  2 = 1 + 1 |   bool            |
     * |    1 + a   |   Option<a>       |
     *
     * 
     * |    Logic   |   Types       |
     * |    ---     |   ---         |
     * |    false   |   Void        |
     * |    true    |   ()          |
     * |    a || b  |   Result<a,b> |
     * |    a && b  |   (a, b)      |
     * 
     * This analogy goes deeper, and is the basis of the Curry-Howard isomorphism between logic and
     * type theory. 
     *
     */

    fn option_to_result<T>(value: Option<T>) -> Result<T, ()> {
        value.map_or(Err(()), |x| Ok(x))
    }
    fn result_to_option<T>(value: Result<T, ()>) -> Option<T> {
        value.ok()
    }

    trait Shape {
        fn area(&self) -> usize;
    }



}


mod chapter_7 {
    pub fn main() {
        let a1 = Some(12);
        let f1 = |x: i32| x > 0;

        dbg!(a1.fmap(f1) == Some(true));
        let a2 = vec![-2, -1, 0, 1, 2];
        dbg!(a2.fmap(f1) == vec![false, false, false, true, true]);
    }

    trait Functor<'a, U, V> {
        type WrapperSelf<T>;
        fn fmap<F: 'a>(self, f: F) -> Self::WrapperSelf<V> where F: Fn(U) -> V;
    }

    impl<U, V> Functor<'_, U, V> for Option<U> {
        type WrapperSelf<T> = Option<T>;
        fn fmap<F>(self, f: F) -> Self::WrapperSelf<V> where F: Fn(U) -> V {
            self.map(f)
        }
    }

    impl<U, V> Functor<'_, U, V> for Vec<U> {
        type WrapperSelf<T> = Vec<T>;
        fn fmap<F>(self, f: F) -> Self::WrapperSelf<V> where F: Fn(U) -> V {
            self.into_iter().map(f).collect()
        }
    }

    impl<'a, U, V: 'a, W> Functor<'a, U, V> for &'a (dyn Fn(W) -> U + 'a) {
        type WrapperSelf<T> = Box<dyn Fn(W) -> V + 'a>;
        fn fmap<F: 'a>(self, f: F) -> Self::WrapperSelf<V> where F: Fn(U) -> V {
            Box::new(move |x| {
                f(self(x))
            })
        }
    }
}

mod chapter_8 {
    pub fn main() {}

    /*
     * bi-functor maps an object from category A and category B to category C 
     *
     */

    trait BiFunctor<T, U, V, W> {
        type WrapperSelf<A, B>;
        fn bimap(self, f: impl Fn(T) -> U, g: impl Fn(V) -> W) -> Self::WrapperSelf<U, W>;
    }

    impl<T, U, V, W> BiFunctor<T, U, V, W> for Result<T, V> {
        type WrapperSelf<A, B> = Result<A, B>;
        fn bimap(self, f: impl Fn(T) -> U, g: impl Fn(V) -> W) -> Self::WrapperSelf<U, W> {
            match self {
                Ok(value) => Ok(f(value)),
                Err(err) => Err(g(err))
            }
        }
    }

    impl<T, U, V, W> BiFunctor<T, U, V, W> for (T, V) {
        type WrapperSelf<A, B> = (A, B);
        fn bimap(self, f: impl Fn(T) -> U, g: impl Fn(V) -> W) -> Self::WrapperSelf<U, W> {
            (f(self.0), g(self.1))
        }
    }
   
    
    /*
     * # Contravariant
     * type signature
     * contramap :: (b -> a) -> f a -> f b
     *
     * The notable observation here is that the map function is the inverse morphism of the one
     * that is being done in the functor space
     *
     * Profunctor signature Functions
     * where p is the Profunctor
     * dimap :: (a -> b) -> (c -> d) -> p b c -> p a d
     * lmap :: (a -> b) -> p b c -> p a c
     * rmap :: (b -> c) -> p a b -> p a c
     *
     * example implmenetation
     * dimap f g = lmap f . rmap g
     * lmap f = dimap f id
     * rmap = dimap id
     *
     * limited by rust type system
     */
}

mod chapter_9 {
    pub fn main() {
        let a = vec![1, 2];

        println!("{:?}", sort_array_by_parity(a));
    }

    
    /*
     * Currying is a function returning a function
     * a -> (b -> c)
     */

    fn curry<'a, T: Clone + 'a, U: 'a, V: 'a>(f: &'a (dyn Fn(T, U) -> V + 'a)) -> Box<dyn Fn(T) -> Box<dyn Fn(U) -> V + 'a> + 'a>
    where
    {
        Box::new(move |x| Box::new(move |y| f(x.clone(), y)))
    }

    fn uncurry<'a, T: 'a, U: 'a, V: 'a>(f: Box<dyn Fn(T) -> Box<dyn Fn(U) -> V + 'a> + 'a>) -> Box<dyn Fn(T, U) -> V + 'a> {
        Box::new(move |x, y| {
            f(x)(y)
        })
    }

    fn sort_array_by_parity(a: Vec<i32>) -> Vec<i32> {
        let plus_plus = Box::new(|mut x: Vec<i32>, y: Vec<i32>| {
            x.extend(y);
            x
        });
        let cplus_plus = curry(&plus_plus);


        let x = partition(|x| x % 2 != 0, a);
        let w = uncurry(cplus_plus)(x.0, x.1);
        w
    }

    fn partition(f: impl Fn(i32) -> bool, data: Vec<i32>) -> (Vec<i32>, Vec<i32>) {
        let output: (Vec<i32>, Vec<i32>) = (vec![], vec![]);

        data.into_iter().fold(output, |mut acc, cur| {
            if f(cur) {
                acc.0.push(cur)
            } else {
                acc.1.push(cur)
            }
            acc
        })
    }
}

mod chapter_10 {
    pub fn main() {}

    
    /*
     * Types of polymorphism
     * Ad-hoc: ~ specific impl
     *  - Function name: Same
     *  - Types: Different
     *  - Behaviour: Different
     * Parametric: ~ impl for generic
     *  - Function name: Same
     *  - Types: Different
     *  - Behaviour: Same
     */


    
    /*
     * natural transformation is a transformation between different functors with the same inner
     * type.
     * Example, Vec -> Option
     */
    fn safe_head<T>(item: impl IntoIterator<Item = T>) -> Option<T> {
        item.into_iter().next()
    }

    
    /*
     * In the case of **Cat** seen as a 2-category we have:
     * - Objects: (Small) categories
     * - 1-morphisms: Functors between categories
     * - 2-morphisms: Natural transformations between functors
     */
}

#[allow(dead_code)]
mod unreachable_universe {
    pub fn main() {
        let a = |x: i32| x + 1;
        let b: Box<dyn Fn(i32) -> i32> = Box::new(a);

        let _v1 = b(1);
        let _v2 = b(2);

    }
}

macro_rules! scope {
    ($f:expr) => {
        println!("-- {} -- [start]", stringify!($f));
        $f();
        println!("-- {} -- [end]", stringify!($f));
    };
}

#[allow(unused_macros)]
macro_rules! nightly_scope {
    ($f:expr) => {
        #[cfg(feature = "nightly-features")]
        scope!($f);
    };
}

fn main() {
    scope!(chapter_1::main);

    scope!(chapter_2::main);

    scope!(chapter_3::main);

    scope!(chapter_7::main);

    scope!(chapter_8::main);

    scope!(chapter_9::main);

    // scope!(unreachable_universe::main);
}
