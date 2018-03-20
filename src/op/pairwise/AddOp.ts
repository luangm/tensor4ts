import PairwiseOp from "./PairwiseOp";

export default class AddOp extends PairwiseOp {

    body(a: number, b: number): number {
        return a + b;
    }

}