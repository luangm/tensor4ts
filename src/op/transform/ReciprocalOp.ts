import TransformOp from "./TransformOp";

export default class ReciprocalOp extends TransformOp {

  body(a: number, b?: number): number {
    return 1 / a;
  }

}