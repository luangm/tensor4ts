import Executor from "./executor/Executor";
import AddOp from "./op/pairwise/AddOp";
import Tensor from "./Tensor";
import ShapeUtils from "./utils/ShapeUtils";

export default class TensorMath {

    static add(left: Tensor, right: Tensor, result?: Tensor): Tensor {
        result = result || Tensor.zeros(ShapeUtils.broadcastShapes(left.shape, right.shape));
        Executor.exec(new AddOp(left, right, result));
        return result;
    }

}