import Operation from "../Operation";

export default abstract class PairwiseOp extends Operation {

    update(accum: number, a: number): number {
        return 0;
    }

    exec(): void {
        // nothing
    }

}