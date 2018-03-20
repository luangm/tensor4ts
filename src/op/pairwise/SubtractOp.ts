import PairwiseOp from "./PairwiseOp";

export default class SubtractOp extends PairwiseOp {

    body(a: number, b: number): number {
        return a - b;
    }

}