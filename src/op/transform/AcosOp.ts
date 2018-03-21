import TransformOp from "./TransformOp";

export default class AcosOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.acos(a);
  }

}